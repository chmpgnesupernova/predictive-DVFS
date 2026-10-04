#ifndef POWER_H
#define POWER_H

/*
 * INA3221 기반 에너지 측정 (독립 power thread, 10ms 주기)
 *
 * 사용법 (main.cpp):
 *   std::thread power_thread;
 *   double consumed_uJ = 0.0;
 *   set_power_measure(&consumed_uJ, &power_thread);   // 측정 시작
 *   ...
 *   get_power_measure(&consumed_uJ, &power_thread);   // 측정 종료, consumed_uJ 에 총 에너지(uJ)
 *
 * 원본 대비 변경점
 *  - 주기: sleep_for(9920us) 누적 drift → sleep_until 고정 10ms 스케줄
 *  - 적분: 직사각형(P_cur * dt) → 사다리꼴 ((P_prev + P_cur)/2 * dt)
 *  - 읽기: FILE* + fseek/fscanf → fd + pread (sysfs 는 매번 offset 0 에서 다시 읽어야 값이 갱신됨)
 *  - 동기화: 측정 중 에너지를 std::atomic 으로 누적 → 다른 쓰레드(vision 요약)에서 안전하게 읽기 가능.
 *            *consumed_uJ 에는 get_power_measure() 시점에 최종값을 기록.
 *  - 헤더 전역변수(debug_cnt, power_thread 등) → inline/static 으로 정리 (ODR)
 *  - 결과 로그: 총 에너지, 측정 시간, 평균 전력, 샘플 수, 실제 샘플링 주기(avg/max), 읽기 실패 수
 *  - 옵션: 샘플별 전력 trace CSV (Log::nowMs() 와 같은 시간축 → dvfs_log.csv 와 정렬 가능)
 *
 * 채널 확인 (Jetson Orin Nano 기준 INA3221 @ 0x40):
 *   cat /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon<N>/in<C>_label
 *   보통 1=VDD_IN(보드 전체), 2=VDD_CPU_GPU_CV, 3=VDD_SOC
 *   hwmon 번호는 부팅/커널에 따라 바뀔 수 있으니 확인 후 POWER_HWMON_DIR 을 맞출 것.
 */

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include <fcntl.h>
#include <unistd.h>

#include "logger.h"

#ifndef POWER_HWMON_DIR
#define POWER_HWMON_DIR "/sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon1"
#endif
#ifndef POWER_CHANNEL
#define POWER_CHANNEL "2"
#endif
#ifndef POWER_PERIOD_US
#define POWER_PERIOD_US 10000          // 10ms
#endif
#ifndef POWER_TRACE_PATH
#define POWER_TRACE_PATH "./power_trace.csv"   // "" 로 빌드하면 trace 비활성화
#endif

namespace power_detail {

inline int volt_fd = -1;
inline int curr_fd = -1;
inline std::atomic<bool> stop_flag{false};
inline std::atomic<double> energy_uJ{0.0};     // 측정 중 누적 에너지 (다른 쓰레드에서 읽기 가능)
inline std::atomic<long> sample_cnt{0};
inline std::atomic<long> read_fail_cnt{0};
inline double start_ms = 0.0;
inline double stop_ms  = 0.0;
inline double period_sum_s = 0.0;
inline double period_max_s = 0.0;
inline bool running = false;
inline std::string trace_path = POWER_TRACE_PATH;   // set_power_trace_path() 로 변경 가능

// sysfs 정수값 읽기 (실패 시 -1)
inline long read_sysfs_long(int fd) {
    if (fd < 0) return -1;
    char buf[32];
    ssize_t n = pread(fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0) return -1;
    buf[n] = '\0';
    char* end = nullptr;
    long v = std::strtol(buf, &end, 10);
    return (end == buf) ? -1 : v;
}

// 현재 전력 (uW = mA * mV). 읽기 실패 시 -1
inline double read_power_uW() {
    long mA = read_sysfs_long(curr_fd);
    long mV = read_sysfs_long(volt_fd);
    if (mA < 0 || mV < 0) return -1.0;
    return static_cast<double>(mA) * static_cast<double>(mV);
}

inline void atomic_add(std::atomic<double>& a, double d) {
    double old = a.load(std::memory_order_relaxed);
    while (!a.compare_exchange_weak(old, old + d, std::memory_order_relaxed)) {}
}

inline void close_fds() {
    if (volt_fd >= 0) { close(volt_fd); volt_fd = -1; }
    if (curr_fd >= 0) { close(curr_fd); curr_fd = -1; }
}

inline void sampling_loop() {
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::microseconds(POWER_PERIOD_US);

    std::ofstream trace;
    if (!trace_path.empty()) {
        trace.open(trace_path, std::ios::out | std::ios::trunc);
        if (trace) trace << "time_ms,power_mW,energy_mJ\n";
    }

    auto last_time = clock::now();
    auto next_wake = last_time + period;
    double prev_uW = read_power_uW();
    if (prev_uW < 0) { prev_uW = 0.0; ++read_fail_cnt; }

    while (!stop_flag.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_until(next_wake);
        next_wake += period;

        auto now = clock::now();
        // 쓰레드가 크게 밀린 경우(스케줄링 지연) 따라잡기 위해 스케줄 재정렬
        if (now > next_wake) next_wake = now + period;

        double dt = std::chrono::duration<double>(now - last_time).count();
        last_time = now;

        double cur_uW = read_power_uW();
        if (cur_uW < 0) {          // 읽기 실패: 직전 값 유지
            ++read_fail_cnt;
            cur_uW = prev_uW;
        }

        atomic_add(energy_uJ, 0.5 * (prev_uW + cur_uW) * dt);   // 사다리꼴 적분
        prev_uW = cur_uW;

        period_sum_s += dt;
        period_max_s = std::max(period_max_s, dt);
        long n = ++sample_cnt;

        if (trace.is_open()) {
            char line[96];
            std::snprintf(line, sizeof(line), "%.3f,%.1f,%.3f\n",
                          Log::nowMs(), cur_uW / 1000.0, energy_uJ.load() / 1000.0);
            trace << line;
            if (n % 100 == 0) trace.flush();   // 1초마다 flush
        }
    }
}

} // namespace power_detail

// trace CSV 경로 변경 (set_power_measure 전에 호출). "" 이면 trace 비활성화
inline void set_power_trace_path(const std::string& path) {
    power_detail::trace_path = path;
}

// 측정 중 누적 에너지(uJ). 다른 쓰레드에서 호출해도 안전 (구간 평균 전력 계산용)
inline double power_energy_now_uJ() {
    return power_detail::energy_uJ.load();
}

// 측정 시작. 실패 시 false (power thread 미생성)
inline bool set_power_measure(double* consumed_uJ, std::thread* power_thread) {
    using namespace power_detail;
    *consumed_uJ = 0.0;
    if (running) {
        Log::event("POWER", "already running");
        return false;
    }

    const std::string volt_path = std::string(POWER_HWMON_DIR) + "/in" POWER_CHANNEL "_input";
    const std::string curr_path = std::string(POWER_HWMON_DIR) + "/curr" POWER_CHANNEL "_input";
    volt_fd = ::open(volt_path.c_str(), O_RDONLY);
    curr_fd = ::open(curr_path.c_str(), O_RDONLY);
    if (volt_fd < 0 || curr_fd < 0) {
        std::cerr << "[Error] Failed to open power node: "
                  << (volt_fd < 0 ? volt_path : curr_path) << std::endl;
        close_fds();
        return false;
    }

    // 채널 label 확인용 로그
    std::string label = "?";
    std::ifstream lf(std::string(POWER_HWMON_DIR) + "/in" POWER_CHANNEL "_label");
    if (lf) std::getline(lf, label);

    energy_uJ.store(0.0);
    sample_cnt.store(0);
    read_fail_cnt.store(0);
    period_sum_s = period_max_s = 0.0;
    stop_flag.store(false);
    start_ms = Log::nowMs();
    running = true;

    *power_thread = std::thread(sampling_loop);

    Log::event("POWER", "start measuring  channel=" POWER_CHANNEL " (" + label + ")  period=" +
               std::to_string(POWER_PERIOD_US / 1000) + "ms  node=" POWER_HWMON_DIR);
    return true;
}

// 측정 종료. *consumed_uJ 에 총 에너지(uJ) 기록 후 결과 로그 출력
inline void get_power_measure(double* consumed_uJ, std::thread* power_thread) {
    using namespace power_detail;
    if (!running) {
        Log::event("POWER", "not running (set_power_measure failed or not called)");
        return;
    }

    stop_flag.store(true);
    if (power_thread->joinable()) power_thread->join();
    stop_ms = Log::nowMs();
    close_fds();
    running = false;

    *consumed_uJ = energy_uJ.load();

    const long   n        = sample_cnt.load();
    const double dur_s    = (stop_ms - start_ms) / 1000.0;
    const double energy_J = *consumed_uJ / 1e6;
    const double avg_mW   = dur_s > 0 ? (*consumed_uJ / dur_s) / 1000.0 : 0.0;

    char buf[320];
    std::snprintf(buf, sizeof(buf),
        "energy=%.3f J (%.0f uJ)  duration=%.2f s  avg_power=%.1f mW  samples=%ld  "
        "period avg=%.3fms max=%.3fms  read_fail=%ld",
        energy_J, *consumed_uJ, dur_s, avg_mW, n,
        n ? period_sum_s / n * 1000.0 : 0.0, period_max_s * 1000.0, read_fail_cnt.load());
    Log::event("POWER", buf);
}

#endif // POWER_H
