#ifndef PREDICTIVE_DVFS_CLASS_H
#define PREDICTIVE_DVFS_CLASS_H

#include <iostream>
#include <fstream>
#include <mutex>
#include <string>

/*
 * class 에 1) hit_count, 2) miss_count 를 도입
 *
 * 1) TAKEN state 의 경우:
 *  miss_count < 20 일 경우 stay,
 *  miss_count >= 20 일 경우 switch
 *
 * 2) NOT_TAKEN state 의 경우:
 *  hit_count < 2 일 경우 stay,
 *  hit_count >= 2 일 경우 switch
 *
 * 3) Middleware 판단 결과는 HIT / MISS / HOLD 3가지:
 *  HIT  : interval 밖에서 target 검출 -> LLM 요청 (interval 시작)
 *  HOLD : interval(1s) 진행 중인 프레임 -> FSM 카운터 변화 없음 (현재 상태 유지)
 *  MISS : interval 밖에서 target 미검출 -> miss 로 카운트
 *  */
// 보드에 맞게 확인할 것: ls /sys/class/devfreq/
#define GPU_DEVFREQ_PATH "/sys/class/devfreq/17000000.gpu"

/*
 * GPU 주파수 제어 (sysfs 직접 쓰기, root 권한 필요 -> sudo ./main 으로 실행)
 *  - 매 프레임(50ms)마다 system("sudo ...") 로 프로세스를 띄우는 비용을 없애기 위해 sysfs 에 직접 쓴다.
 *  - 현재 주파수와 같으면 쓰기를 생략한다.
 *  - hold: LLM 요청이 대기/실행 중이면(hold_ > 0) 주파수를 내리지 않는다.
 *    (추론 도중 miss 프레임의 reactive down-scaling 이 GPU 를 낮추는 것을 방지)
 */
class GpuFreq {
public:
    static constexpr long HIGH_HZ = 624750000;
    static constexpr long LOW_HZ  = 306000000;

    static bool init() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!writeSysfs(GPU_DEVFREQ_PATH "/governor", "userspace")) return false;
        if (!writeSysfs(GPU_DEVFREQ_PATH "/userspace/set_freq", std::to_string(LOW_HZ))) return false;
        cur_hz_ = LOW_HZ;
        return true;
    }

    static void set(long hz) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (hz < cur_hz_ && hold_ > 0) return;   // LLM 작업 중에는 down-scaling 무시
        setLocked(hz);
    }

    // LLM 요청 1건이 버퍼에 들어갈 때 호출
    static void acquire() {
        std::lock_guard<std::mutex> lock(mtx_);
        ++hold_;
    }

    // LLM 요청 1건이 끝날 때 호출. 남은 요청이 없으면 LOW 로 내린다.
    static void release() {
        std::lock_guard<std::mutex> lock(mtx_);
        if (hold_ > 0 && --hold_ == 0) setLocked(LOW_HZ);
    }

    // 비정상 종료 시 hold 해제
    static void releaseAll() {
        std::lock_guard<std::mutex> lock(mtx_);
        hold_ = 0;
        setLocked(LOW_HZ);
    }

private:
    static void setLocked(long hz) {
        if (hz == cur_hz_) return;
        if (writeSysfs(GPU_DEVFREQ_PATH "/userspace/set_freq", std::to_string(hz))) {
            cur_hz_ = hz;
        }
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
    inline static int hold_ = 0;
};


// State 는 2 state 로 변경
enum FSMState {
    TAKEN,
    NOT_TAKEN
};

// Middleware 판단 결과
enum class Decision {
    HIT,    // LLM 요청
    MISS,   // 요청 없음 (interval 밖)
    HOLD    // interval 진행 중 → 카운트하지 않음
};

inline const char* decisionToString(Decision d) {
    switch (d) {
        case Decision::HIT:  return "HIT";
        case Decision::MISS: return "MISS";
        case Decision::HOLD: return "HOLD";
    }
    return "UNKNOWN";
}

class PredictiveDVFS {
private:
    static constexpr int MISS_THRESHOLD = 20;
    static constexpr int HIT_THRESHOLD  = 2;

    FSMState state = TAKEN;
    std::mutex mtx;
    bool early_scale = false;

    int miss_count = 0;
    int hit_count = 0;

    void update(Decision d) {
        if (d == Decision::HOLD) return;   // interval 중: 카운터·상태 유지

        const bool is_hit = (d == Decision::HIT);
        if (state == TAKEN) {
            if (is_hit) {
                miss_count = 0;
            } else if (++miss_count >= MISS_THRESHOLD) {
                state = NOT_TAKEN;
                miss_count = 0;          // 전이 시 카운터 초기화
                hit_count = 0;
            }
        } else { // NOT_TAKEN
            if (!is_hit) {
                hit_count = 0;
            } else if (++hit_count >= HIT_THRESHOLD) {
                state = TAKEN;
                hit_count = 0;           // 전이 시 카운터 초기화
                miss_count = 0;
            }
        }
    }

    std::string stateToString(FSMState s) {
        switch (s) {
            case TAKEN:         return "TAKEN";
            case NOT_TAKEN:     return "NOT_TAKEN";
            default:            return "UNKNOWN";
        }
    }

public:
    FSMState getState() {
        std::lock_guard<std::mutex> lock(mtx);
        return state;
    }

    std::string getStateString() {
        std::lock_guard<std::mutex> lock(mtx);
        return stateToString(state);
    }

    void scaleFrequencyEarly() {
        std::lock_guard<std::mutex> lock(mtx);
        if (state == TAKEN) {
            // DVFS start, scale freq
            GpuFreq::set(GpuFreq::HIGH_HZ);
            early_scale = true;
        } else {
            // already low freq
            early_scale = false;
        }
    }

    void scaleFrequencyReactive(Decision d) {
        std::lock_guard<std::mutex> lock(mtx);

        if (!early_scale && d == Decision::HIT) {
            // DVFS start, scale freq
            GpuFreq::set(GpuFreq::HIGH_HZ);
        } else if (early_scale && d != Decision::HIT) {
            // False alarm (MISS) 또는 interval 중(HOLD): 요청이 없으므로 down freq
            // (LLM 작업이 대기/진행 중이면 GpuFreq 가 무시)
            GpuFreq::set(GpuFreq::LOW_HZ);
        }
        // early_scale && HIT, !early_scale && (MISS|HOLD) 는 skip.

        update(d);
    }
};

#endif
