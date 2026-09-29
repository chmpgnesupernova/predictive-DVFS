#ifndef PREDICTIVE_DVFS_CLASS_H
#define PREDICTIVE_DVFS_CLASS_H

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <mutex>
#include <string>

#include "logger.h"

/*
 * Predictive-DVFS Controller
 *
 * hit / miss 정의: robotMiddleware 가 해당 프레임에 LLM request 를 했으면 HIT, 안 했으면 MISS.
 * miss_count 는 LLM request(HIT) 직후 프레임부터 센다.
 * (1초에 1번 request 하는 환경 → HIT 1회 + MISS 19회 반복)
 *
 * 1) TAKEN state:
 *    HIT                     : miss_count = 0, stay
 *    MISS, miss_count < 25   : stay
 *    MISS, miss_count >= 25  : → NOT_TAKEN   (1.25초 동안 request 없음)
 *
 * 2) NOT_TAKEN state:
 *    HIT, hit_count < 2      : stay
 *    HIT, hit_count >= 2     : → TAKEN
 *    MISS                    : stay. 단, 마지막 HIT 이후 miss_count >= 25 가 되면 hit_count = 0
 *                              (= HIT 사이 간격이 25 프레임 이내일 때만 "연속 HIT" 로 인정.
 *                                 MISS 한 번에 hit_count 를 0 으로 하면 1초 주기 환경에서
 *                                 NOT_TAKEN → TAKEN 전이가 불가능하므로)
 */

// 보드에 맞게 확인할 것: ls /sys/class/devfreq/
#ifndef GPU_DEVFREQ_PATH
#define GPU_DEVFREQ_PATH "/sys/class/devfreq/170000.gpu"
#endif

// 주파수 변경 원인
enum class FreqReason {
    INIT,              // 시작 시 LOW 로 초기화
    EARLY_UP,          // TAKEN: 비전 추론 시작과 동시에 선제적 up        (매 프레임 반복 → 파일만)
    REACTIVE_UP,       // NOT_TAKEN 에서 HIT: 사후 up
    FALSE_ALARM_DOWN,  // TAKEN 에서 MISS: 선제 up 을 되돌림             (매 프레임 반복 → 파일만)
    LLM_END_DOWN,      // LLM_END_MARKER 수신 즉시 down
    SHUTDOWN           // 종료 시 LOW
};

inline const char* reasonToString(FreqReason r) {
    switch (r) {
        case FreqReason::INIT:             return "INIT";
        case FreqReason::EARLY_UP:         return "EARLY_UP";
        case FreqReason::REACTIVE_UP:      return "REACTIVE_UP";
        case FreqReason::FALSE_ALARM_DOWN: return "FALSE_ALARM_DOWN";
        case FreqReason::LLM_END_DOWN:     return "LLM_END_DOWN";
        case FreqReason::SHUTDOWN:         return "SHUTDOWN";
    }
    return "UNKNOWN";
}

// 매 프레임 반복되는 이벤트는 콘솔에서 제외 (CSV 에는 모두 기록, 콘솔은 100 프레임 요약에 횟수로 표시)
inline bool isPeriodicReason(FreqReason r) {
    return r == FreqReason::EARLY_UP || r == FreqReason::FALSE_ALARM_DOWN;
}

/*
 * GPU 주파수 제어 (sysfs 직접 쓰기, root 권한 필요 -> sudo ./main 으로 실행)
 *  - 현재 주파수와 같으면 쓰기를 생략한다.
 *  - pending(hold): LLM 요청이 대기/실행 중이면 down-scaling 을 막는다.
 *    (LLM 추론 중 MISS 프레임의 FALSE_ALARM_DOWN 이 GPU 를 낮추는 것을 방지)
 *  - LLM_END_MARKER 수신 즉시 release() → 남은 요청이 없으면 곧바로 LOW.
 */
// 주파수 변경 횟수 카운터
struct FreqCounters {
    int up = 0;
    int down = 0;
    int early_up = 0;
    int false_alarm_down = 0;
};

class GpuFreq {
public:
    static constexpr long HIGH_HZ = 624750000;
    static constexpr long LOW_HZ  = 306000000;

    using Counters = FreqCounters;

    static bool init() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!writeSysfs(GPU_DEVFREQ_PATH "/governor", "userspace")) return false;
        cur_hz_ = -1;
        setLocked(LOW_HZ, FreqReason::INIT);
        return cur_hz_ == LOW_HZ;
    }

    static void set(long hz, FreqReason reason) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (hz < cur_hz_ && pending_ > 0) {
            if (DVFS_VERBOSE) {
                Log::fileOnly("FREQ_BLOCKED", std::string(reasonToString(reason)) +
                              " blocked (LLM pending=" + std::to_string(pending_) + ")");
            }
            return;
        }
        setLocked(hz, reason);
    }

    // LLM 요청 1건이 버퍼에 들어가기 직전에 호출
    static void acquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++pending_;
    }

    // push 가 새 항목을 만들지 못한 경우(overwrite/closed) acquire 취소
    static void cancelAcquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pending_ > 0) --pending_;
    }

    // LLM_END_MARKER 수신 즉시 호출. 남은 요청이 없으면 LOW 로 내린다.
    static void release() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (pending_ > 0) --pending_;
        if (pending_ == 0) {
            setLocked(LOW_HZ, FreqReason::LLM_END_DOWN);
        } else {
            Log::event("FREQ", "LLM_END but " + std::to_string(pending_) +
                       " request(s) pending in ring buffer -> keep " + mhz(cur_hz_));
        }
    }

    static void releaseAll() {
        std::lock_guard<std::mutex> lock(mtx_);
        pending_ = 0;
        setLocked(LOW_HZ, FreqReason::SHUTDOWN);
    }

    static long current() {
        std::lock_guard<std::mutex> lock(mtx_);
        return cur_hz_;
    }

    // 요약 출력용: 구간 카운터를 가져오고 초기화
    static Counters takeWindowCounters() {
        std::lock_guard<std::mutex> lock(mtx_);
        Counters c = window_;
        window_ = Counters{};
        return c;
    }

    static Counters totalCounters() {
        std::lock_guard<std::mutex> lock(mtx_);
        return total_;
    }

    static std::string mhz(long hz) {
        if (hz <= 0) return "unknown";
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2fMHz", hz / 1e6);
        return buf;
    }

private:
    static void setLocked(long hz, FreqReason reason) {
        if (hz == cur_hz_) return;

        double t = Log::nowMs();
        auto w0 = std::chrono::steady_clock::now();
        bool ok = writeSysfs(GPU_DEVFREQ_PATH "/userspace/set_freq", std::to_string(hz));
        double write_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - w0).count();
        if (!ok) return;

        const bool up = hz > cur_hz_;
        const std::string prev = mhz(cur_hz_);
        cur_hz_ = hz;

        // 카운터
        for (Counters* c : {&window_, &total_}) {
            if (up) ++c->up; else ++c->down;
            if (reason == FreqReason::EARLY_UP)         ++c->early_up;
            if (reason == FreqReason::FALSE_ALARM_DOWN) ++c->false_alarm_down;
        }

        char buf[160];
        std::snprintf(buf, sizeof(buf), "%s %s -> %s  reason=%s  sysfs_write=%.0fus",
                      up ? "UP  " : "DOWN", prev.c_str(), mhz(hz).c_str(),
                      reasonToString(reason), write_us);

        if (isPeriodicReason(reason)) Log::fileOnlyAt(t, "FREQ", buf);
        else                          Log::eventAt(t, "FREQ", buf);
    }

    static bool writeSysfs(const std::string& path, const std::string& value) {
        std::ofstream f(path);
        if (!f) {
            std::cerr << "[DVFS] cannot open " << path << " (root 권한/경로 확인)\n";
            return false;
        }
        f << value;
        f.flush();
        if (!f) {
            std::cerr << "[DVFS] failed to write '" << value << "' to " << path << "\n";
            return false;
        }
        return true;
    }

    inline static std::mutex mtx_;
    inline static long cur_hz_ = -1;
    inline static int pending_ = 0;
    inline static Counters window_{};
    inline static Counters total_{};
};


enum FSMState {
    TAKEN,
    NOT_TAKEN
};

inline const char* stateToString(FSMState s) {
    switch (s) {
        case TAKEN:     return "TAKEN";
        case NOT_TAKEN: return "NOT_TAKEN";
    }
    return "UNKNOWN";
}

class PredictiveDVFS {
public:
    static constexpr int MISS_THRESHOLD = 25;
    static constexpr int HIT_THRESHOLD  = 2;

    struct Snapshot {
        FSMState state;
        int miss_count;
        int hit_count;
        int transitions;
    };

    // 비전 추론 직전에 호출
    void scaleFrequencyEarly() {
        std::lock_guard<std::mutex> lock(mtx);
        if (state == TAKEN) {
            early_t_ms = Log::nowMs();
            GpuFreq::set(GpuFreq::HIGH_HZ, FreqReason::EARLY_UP);
            early_scale = true;
        } else {
            early_scale = false;
        }
    }

    // middleware 판단 직후 호출 (is_hit = LLM request 여부)
    void scaleFrequencyReactive(bool is_hit) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!early_scale && is_hit) {
            GpuFreq::set(GpuFreq::HIGH_HZ, FreqReason::REACTIVE_UP);
        } else if (early_scale && !is_hit) {
            // False alarm: 선제 up 되돌림 (LLM 대기/실행 중이면 GpuFreq 가 무시)
            GpuFreq::set(GpuFreq::LOW_HZ, FreqReason::FALSE_ALARM_DOWN);
        }
        // early_scale && HIT, !early_scale && MISS 는 skip.

        if (is_hit) {
            const int miss_before = miss_count;
            const int hit_before  = hit_count;
            const FSMState state_before = state;
            update(true);

            char buf[200];
            if (early_scale) {
                std::snprintf(buf, sizeof(buf),
                    "HIT  state=%s%s%s scaling=EARLY(lead %.1fms before request)  miss_count %d->%d  hit_count %d->%d",
                    stateToString(state_before), state_before != state ? "->" : "",
                    state_before != state ? stateToString(state) : "", Log::nowMs() - early_t_ms,
                    miss_before, miss_count, hit_before, hit_count);
            } else {
                std::snprintf(buf, sizeof(buf),
                    "HIT  state=%s%s%s scaling=REACTIVE  miss_count %d->%d  hit_count %d->%d",
                    stateToString(state_before), state_before != state ? "->" : "",
                    state_before != state ? stateToString(state) : "", miss_before, miss_count, hit_before, hit_count);
            }
            Log::event("FSM", buf);
        } else {
            update(false);
            if (DVFS_VERBOSE) {
                Log::fileOnly("FSM", std::string("MISS state=") + stateToString(state) +
                              " miss_count=" + std::to_string(miss_count) +
                              " hit_count=" + std::to_string(hit_count));
            }
        }
    }

    Snapshot snapshot() {
        std::lock_guard<std::mutex> lock(mtx);
        return {state, miss_count, hit_count, transitions};
    }

private:
    FSMState state = TAKEN;
    std::mutex mtx;
    bool early_scale = false;
    double early_t_ms = 0.0;

    int miss_count = 0;
    int hit_count = 0;
    int transitions = 0;

    void transition(FSMState next, const char* why) {
        char buf[160];
        std::snprintf(buf, sizeof(buf), "STATE %s -> %s  (%s, miss_count=%d, hit_count=%d)",
                      stateToString(state), stateToString(next), why, miss_count, hit_count);
        Log::event("FSM", buf);
        state = next;
        miss_count = 0;
        hit_count = 0;
        ++transitions;
    }

    void update(bool is_hit) {
        if (state == TAKEN) {
            if (is_hit) {
                miss_count = 0;
            } else if (++miss_count >= MISS_THRESHOLD) {
                transition(NOT_TAKEN, "miss_count >= 25");
            }
        } else { // NOT_TAKEN
            if (is_hit) {
                miss_count = 0;
                if (++hit_count >= HIT_THRESHOLD) {
                    transition(TAKEN, "hit_count >= 2");
                }
            } else if (miss_count < MISS_THRESHOLD) {
                // 마지막 HIT 이후 25 프레임 동안 HIT 가 없으면 연속성 끊김
                if (++miss_count >= MISS_THRESHOLD && hit_count > 0) {
                    if (DVFS_VERBOSE) {
                        Log::fileOnly("FSM", "NOT_TAKEN hit_count " + std::to_string(hit_count) +
                                      " -> 0 (no HIT for 25 frames)");
                    }
                    hit_count = 0;
                }
            }
            // miss_count 는 MISS_THRESHOLD 에서 포화 (NOT_TAKEN 에서는 의미 없음)
        }
    }
};

#endif
