#ifndef VLA_LOGGER_H
#define VLA_LOGGER_H

#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <string>

/*
 * 실험 로그 유틸
 *  - 모든 시간은 프로그램 시작 기준 상대시간(ms, steady_clock)
 *  - Log::event()   : 콘솔 + CSV 파일
 *  - Log::fileOnly(): CSV 파일에만 기록 (매 프레임 반복되는 이벤트용)
 *
 * CSV 컬럼: time_ms,type,detail
 *
 * DVFS_VERBOSE=1 로 빌드하면 fileOnly 이벤트도 콘솔에 출력 (디버깅용)
 *   g++ ... -DDVFS_VERBOSE=1
 */
#ifndef DVFS_VERBOSE
#define DVFS_VERBOSE 0
#endif

#ifndef DVFS_LOG_PATH
#define DVFS_LOG_PATH "./dvfs_log.csv"
#endif

namespace Log {

inline const std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
inline std::mutex mtx;
inline std::ofstream csv;

inline double nowMs() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

inline bool open(const std::string& path = DVFS_LOG_PATH) {
    std::lock_guard<std::mutex> lock(mtx);
    csv.open(path, std::ios::out | std::ios::trunc);
    if (!csv) return false;
    csv << "time_ms,type,detail\n";
    return true;
}

inline void close() {
    std::lock_guard<std::mutex> lock(mtx);
    if (csv.is_open()) csv.close();
}

inline std::string csvEscape(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out + "\"";
}

inline void write(double t, const std::string& type, const std::string& detail, bool console) {
    std::lock_guard<std::mutex> lock(mtx);
    if (console) {
        std::printf("[%10.2f ms] [%-12s] %s\n", t, type.c_str(), detail.c_str());
        std::fflush(stdout);
    }
    if (csv.is_open()) {
        char tbuf[32];
        std::snprintf(tbuf, sizeof(tbuf), "%.3f", t);
        csv << tbuf << ',' << type << ',' << csvEscape(detail) << '\n';
    }
}

inline void event(const std::string& type, const std::string& detail) {
    write(nowMs(), type, detail, true);
}

inline void eventAt(double t, const std::string& type, const std::string& detail) {
    write(t, type, detail, true);
}

inline void fileOnly(const std::string& type, const std::string& detail) {
    write(nowMs(), type, detail, DVFS_VERBOSE != 0);
}

inline void fileOnlyAt(double t, const std::string& type, const std::string& detail) {
    write(t, type, detail, DVFS_VERBOSE != 0);
}

// 여러 줄 블록(LLM output 등)을 콘솔에 그대로 출력 + CSV 1행
inline void block(const std::string& type, const std::string& header, const std::string& body) {
    double t = nowMs();
    std::lock_guard<std::mutex> lock(mtx);
    std::printf("[%10.2f ms] [%-12s] %s\n", t, type.c_str(), header.c_str());
    std::printf("---------------- %s ----------------\n%s\n-----------------------------------------\n",
                type.c_str(), body.c_str());
    std::fflush(stdout);
    if (csv.is_open()) {
        char tbuf[32];
        std::snprintf(tbuf, sizeof(tbuf), "%.3f", t);
        csv << tbuf << ',' << type << ',' << csvEscape(header + " | " + body) << '\n';
    }
}

} // namespace Log

#endif // VLA_LOGGER_H
