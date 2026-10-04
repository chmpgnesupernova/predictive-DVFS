#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <csignal>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "vision.h"
#include "logger.h"
#include "dvfs_class.h"
#include "ring_buffer.h"
#include "power.h"

#ifndef TARGET_FPS
#define TARGET_FPS 20
#endif
#ifndef TIME_SLOT_SEC
#define TIME_SLOT_SEC 60
#endif
#ifndef TIME_SLOT_NUM
#define TIME_SLOT_NUM 10
#endif
#define RING_BUFFER_SIZE 8
#define REQUEST_INTERVAL_MS 1000      // LLM request 주기 (1초에 1번)
#define VISION_LOG_EVERY 100          // vision 요약 로그 주기 (100 프레임 = 5초)

// LLM 자식 프로세스 설정
#ifndef MOTION_PLAN_BIN
#define MOTION_PLAN_BIN "./motion-plan"
#endif
#ifndef LLM_MODEL_PATH
#define LLM_MODEL_PATH "./models/Llama-3.2-1B-Instruction-Q4_K_M.gguf"
#endif

// motion-plan.cpp 와 동일해야 하는 프로토콜 마커
static const char* LLM_READY_MARKER = "<<MOTION_PLAN_READY>>";
static const char* LLM_END_MARKER   = "<<MOTION_PLAN_END>>";

// 슬롯 별 테스트용 프레임 이미지 리스트
std::vector<std::string> image_slot = {
    "./images/orange.jpg", "./images/apple.jpg", "./images/orange.jpg", "./images/apple.jpg",
    "./images/orange.jpg", "./images/apple.jpg", "./images/orange.jpg", "./images/apple.jpg",
    "./images/orange.jpg", "./images/apple.jpg"
};

std::map<int, std::string> class_map = {
    {957, "apple"},
    {950, "orange"}
};

// Vision -> LLM 요청 1건
struct LlmRequest {
    int id = 0;
    int frame = 0;
    double push_ms = 0.0;      // ring buffer push 시각
    std::string input;         // vision 결과
};

// global
DvfsMode g_mode = DvfsMode::PREDICTIVE;
PredictiveDVFS dvfsController;
RingBuffer<LlmRequest, RING_BUFFER_SIZE> vision_ring;

// LLM 통계 (LLM thread 에서만 갱신, 종료 후 main 에서 읽음)
struct LlmStats {
    int count = 0;
    double sum_ms = 0.0, min_ms = 1e18, max_ms = 0.0;
    double sum_wait_ms = 0.0;
} llm_stats;

// ---------------------------------------------------------------------------
// LLM 자식 프로세스 (motion-plan) 래퍼
//   parent --(stdin pipe)--> motion-plan : 한 줄 = 요청 1건 (vision 결과)
//   parent <--(stdout pipe)-- motion-plan : 응답 ... LLM_END_MARKER
// ---------------------------------------------------------------------------
class LlmProcess {
public:
    bool start(const std::vector<std::string>& argv) {
        int in_pipe[2], out_pipe[2];
        if (pipe(in_pipe) != 0 || pipe(out_pipe) != 0) {
            perror("[LLM] pipe");
            return false;
        }

        std::vector<char*> args;
        for (const auto& s : argv) args.push_back(const_cast<char*>(s.c_str()));
        args.push_back(nullptr);

        pid_ = fork();
        if (pid_ < 0) {
            perror("[LLM] fork");
            return false;
        }

        if (pid_ == 0) {
            // child: stdin/stdout 을 pipe 로 연결 (stderr 는 터미널 그대로)
            dup2(in_pipe[0], STDIN_FILENO);
            dup2(out_pipe[1], STDOUT_FILENO);
            close(in_pipe[0]);  close(in_pipe[1]);
            close(out_pipe[0]); close(out_pipe[1]);
            execv(args[0], args.data());
            perror("[LLM] execv");
            _exit(127);
        }

        close(in_pipe[0]);
        close(out_pipe[1]);
        to_child_   = fdopen(in_pipe[1], "w");
        from_child_ = fdopen(out_pipe[0], "r");
        if (!to_child_ || !from_child_) {
            perror("[LLM] fdopen");
            return false;
        }

        Log::event("LLM", "waiting for motion-plan to load model...");
        double t0 = Log::nowMs();
        bool ok = readUntil(LLM_READY_MARKER, nullptr, nullptr);
        if (ok) {
            char buf[96];
            std::snprintf(buf, sizeof(buf), "motion-plan ready (model load %.0f ms)", Log::nowMs() - t0);
            Log::event("LLM", buf);
        }
        return ok;
    }

    // 요청 1건 전송 후 LLM_END_MARKER 까지 block.
    // on_end 는 마커를 읽은 즉시(응답 후처리 전에) 호출된다.
    bool request(const std::string& input, std::string& response, const std::function<void()>& on_end) {
        if (fprintf(to_child_, "%s\n", input.c_str()) < 0 || fflush(to_child_) != 0) {
            return false;
        }
        return readUntil(LLM_END_MARKER, &response, &on_end);
    }

    void stop() {
        if (to_child_)   { fclose(to_child_);   to_child_ = nullptr; }   // EOF -> child 종료
        if (from_child_) {
            char buf[256];
            while (fgets(buf, sizeof(buf), from_child_)) {}
            fclose(from_child_);
            from_child_ = nullptr;
        }
        if (pid_ > 0) {
            int status = 0;
            waitpid(pid_, &status, 0);
            pid_ = -1;
        }
    }

private:
    bool readUntil(const std::string& marker, std::string* out, const std::function<void()>* on_end) {
        char* line = nullptr;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, from_child_)) != -1) {
            std::string s(line, n);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();

            if (s == marker) {
                if (on_end && *on_end) (*on_end)();   // 마커 수신 즉시 (GPU down)
                free(line);
                return true;
            }
            if (out) { *out += s; *out += '\n'; }
        }
        free(line);
        return false;   // child 종료 (EOF)
    }

    pid_t pid_ = -1;
    FILE* to_child_ = nullptr;
    FILE* from_child_ = nullptr;
};

// ---------------------------------------------------------------------------
// Robot Middleware
//   1초에 1번 LLM request 를 발생시킨다 (target object 가 보일 때).
//   request 한 프레임 = HIT, 나머지 프레임 = MISS.
// ---------------------------------------------------------------------------
static std::chrono::steady_clock::time_point last_request_time;
static bool has_requested = false;

bool robotMiddleware(const std::string& vision_output) {
    auto now = std::chrono::steady_clock::now();

    if (has_requested) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_request_time).count();
        if (elapsed < REQUEST_INTERVAL_MS) return false;
    }

    if (vision_output == "apple") {
        last_request_time = now;
        has_requested = true;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// LLM Thread
// ---------------------------------------------------------------------------
static std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    size_t e = s.find_last_not_of(" \t\r\n");
    return (b == std::string::npos) ? "" : s.substr(b, e - b + 1);
}

void llmThreadFunc(LlmProcess& llm) {
    LlmRequest req;

    // close() 후 버퍼가 비면 pop 이 false 를 반환하며 종료
    while (vision_ring.pop(req)) {
        const double start_ms = Log::nowMs();
        const double wait_ms  = start_ms - req.push_ms;

        char buf[200];
        std::snprintf(buf, sizeof(buf), "START #%d input='%s' frame=%d queue_wait=%.1fms gpu=%s",
                      req.id, req.input.c_str(), req.frame, wait_ms,
                      GpuFreq::mhz(GpuFreq::current()).c_str());
        Log::eventAt(start_ms, "LLM", buf);

        double end_ms = 0.0;
        std::string response;
        bool ok = llm.request(req.input, response, [&] {
            end_ms = Log::nowMs();
            GpuFreq::release();          // LLM_END_MARKER 수신 즉시 GPU LOW
        });

        if (!ok) {
            Log::event("ERROR", "motion-plan process terminated unexpectedly");
            vision_ring.close();
            GpuFreq::releaseAll();
            break;
        }

        const double llm_ms   = end_ms - start_ms;           // LLM 동작 시간
        const double total_ms = end_ms - req.push_ms;        // request -> 응답 완료

        llm_stats.count++;
        llm_stats.sum_ms += llm_ms;
        llm_stats.min_ms = std::min(llm_stats.min_ms, llm_ms);
        llm_stats.max_ms = std::max(llm_stats.max_ms, llm_ms);
        llm_stats.sum_wait_ms += wait_ms;

        std::snprintf(buf, sizeof(buf), "END   #%d llm_time=%.1fms (request->end %.1fms)",
                      req.id, llm_ms, total_ms);
        Log::eventAt(end_ms, "LLM", buf);

        Log::block("LLM_OUTPUT", "#" + std::to_string(req.id) + " '" + req.input + "'", trim(response));
    }
}

// ---------------------------------------------------------------------------
// Vision Thread
// ---------------------------------------------------------------------------
void visionThreadFunc() {
    const int frame_duration_ms = 1000 / TARGET_FPS; // 50ms
    const int total_frame = TIME_SLOT_NUM * TIME_SLOT_SEC * TARGET_FPS;
    const int frames_per_slot = TIME_SLOT_SEC * TARGET_FPS;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisionInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(6); // Jetson CPU 코어 최적화

    const std::string model_path = "./models/mobilenetv2.onnx";
    const char* input_node_name = "input";
    const char* output_node_name = "output";

    Ort::Session session(env, model_path.c_str(), session_options);

    // 100 프레임 요약용
    double win_lat_sum = 0.0, win_lat_max = 0.0;
    double win_energy_start_uJ = power_energy_now_uJ();
    double win_start_ms = Log::nowMs();
    int win_frames = 0, win_requests = 0, win_overruns = 0;
    std::string last_detect;
    float last_score = 0.0f;
    int request_id = 0;

    for (int frame_count = 0; frame_count < total_frame; ++frame_count) {
        auto start_time = std::chrono::steady_clock::now();

        int cur_slot = frame_count / frames_per_slot;
        const std::string& img = image_slot[cur_slot % image_slot.size()];
        if (frame_count % frames_per_slot == 0) {
            Log::event("SLOT", "slot " + std::to_string(cur_slot) + " start: " + img +
                       " (frame " + std::to_string(frame_count) + ")");
        }

        // Early Scaling
        dvfsController.scaleFrequencyEarly();

        InferenceResult result = runMobileNetInference(
            img, session, input_node_name, output_node_name
        );

        std::string vision_output;
        if (result.class_id == -1) {
            vision_output = "none"; // 이미지 로드 실패 등 예외 처리
        } else {
            auto it = class_map.find(result.class_id);
            vision_output = (it != class_map.end()) ? it->second
                                                    : "object_" + std::to_string(result.class_id);
        }
        last_detect = vision_output;
        last_score = result.confidence;
        win_lat_sum += result.inference_time_ms;
        win_lat_max = std::max(win_lat_max, result.inference_time_ms);

        // Middleware 판단 (request 했으면 HIT, 아니면 MISS)
        bool is_hit = robotMiddleware(vision_output);

        // Reactive Scaling & FSM 업데이트
        dvfsController.scaleFrequencyReactive(is_hit);

        // Ring buffer 를 이용한 LLM 데이터 전달
        if (is_hit) {
            ++win_requests;
            LlmRequest req{++request_id, frame_count, Log::nowMs(), vision_output};
            GpuFreq::acquire();                     // 응답 완료 전까지 GPU down-scaling 방지
            PushResult r = vision_ring.push(req);
            if (r != PushResult::Accepted) {
                GpuFreq::cancelAcquire();
                if (r == PushResult::Overwrote) {
                    Log::event("RING", "buffer full, oldest request dropped (request #" +
                               std::to_string(req.id) + ")");
                }
            }
        }

        // 20FPS 유지를 위한 보정
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed > frame_duration_ms) ++win_overruns;
        ++win_frames;

        // 100 프레임(5초)마다 1회 요약
        if ((frame_count + 1) % VISION_LOG_EVERY == 0 || frame_count + 1 == total_frame) {
            auto s = dvfsController.snapshot();
            auto c = GpuFreq::takeWindowCounters();
            char fsm_buf[80];
            if (g_mode == DvfsMode::PREDICTIVE) {
                std::snprintf(fsm_buf, sizeof(fsm_buf), "state=%s miss_count=%d hit_count=%d",
                              stateToString(s.state), s.miss_count, s.hit_count);
            } else {
                std::snprintf(fsm_buf, sizeof(fsm_buf), "mode=%s", modeToString(g_mode));
            }
            const double now_ms = Log::nowMs();
            const double now_uJ = power_energy_now_uJ();
            const double win_power_mW = (now_ms > win_start_ms)
                ? (now_uJ - win_energy_start_uJ) / (now_ms - win_start_ms) : 0.0;   // uJ/ms = mW
            win_energy_start_uJ = now_uJ;
            win_start_ms = now_ms;
            char buf[440];
            std::snprintf(buf, sizeof(buf),
                "frames %d-%d  detect='%s'(%.2f)  vision_lat avg=%.1fms max=%.1fms  overrun=%d  "
                "requests=%d  | %s  "
                "| freq up=%d(early %d) down=%d(false_alarm %d) gpu=%s  ring=%zu  | power avg=%.1fmW",
                frame_count + 1 - win_frames, frame_count, last_detect.c_str(), last_score,
                win_lat_sum / win_frames, win_lat_max, win_overruns, win_requests,
                fsm_buf,
                c.up, c.early_up, c.down, c.false_alarm_down,
                GpuFreq::mhz(GpuFreq::current()).c_str(), vision_ring.size(), win_power_mW);
            Log::event("VISION", buf);
            win_lat_sum = win_lat_max = 0.0;
            win_frames = win_requests = win_overruns = 0;
        }

        if (elapsed < frame_duration_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_duration_ms - elapsed));
        }
    }

    vision_ring.close();
}

// "./dvfs_log.csv" + "max_freq" -> "./dvfs_log_max_freq.csv"
static std::string withModeSuffix(const std::string& path, DvfsMode mode) {
    if (path.empty()) return path;
    size_t slash = path.find_last_of('/');
    size_t dot = path.find_last_of('.');
    std::string suffix = std::string("_") + modeToString(mode);
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash)) return path + suffix;
    return path.substr(0, dot) + suffix + path.substr(dot);
}

static void printUsage(const char* prog) {
    std::fprintf(stderr,
        "usage: sudo %s [--mode predictive|max_freq|reactive_only]\n"
        "  predictive    : Predictive-DVFS (default)\n"
        "  max_freq      : baseline 1, GPU fixed at max frequency\n"
        "  reactive_only : baseline 2, no early scaling (up on LLM request, down on LLM end)\n",
        prog);
}

int main(int argc, char** argv) {
    // 자식 프로세스가 죽은 뒤 pipe 에 쓰면 SIGPIPE 로 전체가 종료되는 것을 방지
    signal(SIGPIPE, SIG_IGN);

    // 실행 모드 파싱
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--mode" || a == "-m") && i + 1 < argc) {
            if (!parseMode(argv[++i], g_mode)) { printUsage(argv[0]); return 1; }
        } else if (a.rfind("--mode=", 0) == 0) {
            if (!parseMode(a.substr(7), g_mode)) { printUsage(argv[0]); return 1; }
        } else if (a == "-h" || a == "--help") {
            printUsage(argv[0]); return 0;
        } else {
            printUsage(argv[0]); return 1;
        }
    }

    // 모드별 로그 파일 분리 (결과 덮어쓰기 방지)
    const std::string log_path   = withModeSuffix(DVFS_LOG_PATH, g_mode);
    const std::string trace_path = withModeSuffix(POWER_TRACE_PATH, g_mode);
    set_power_trace_path(trace_path);

    if (!Log::open(log_path)) {
        std::cerr << "cannot open log file " << log_path << "\n";
        return 1;
    }
    Log::event("SYSTEM", std::string("Starting VLA Predictive-DVFS Framework  mode=") + modeToString(g_mode) +
               "  (MISS_THRESHOLD=" + std::to_string(PredictiveDVFS::MISS_THRESHOLD) +
               ", HIT_THRESHOLD=" + std::to_string(PredictiveDVFS::HIT_THRESHOLD) +
               ")  log=" + log_path + "  power_trace=" + (trace_path.empty() ? "off" : trace_path));

    dvfsController.setMode(g_mode);
    if (!GpuFreq::init(g_mode)) {
        std::cerr << "GPU DVFS init failed. Run with sudo and check " GPU_DEVFREQ_PATH "\n";
        return 1;
    }

    // 쓰레드 생성 전에 fork (멀티쓰레드 상태에서의 fork 회피) + 모델 로드 완료 대기
    LlmProcess llm;
    if (!llm.start({MOTION_PLAN_BIN, "-m", LLM_MODEL_PATH})) {
        std::cerr << "Failed to start motion-plan process.\n";
        llm.stop();
        return 1;
    }

    // power start (모델 로드 이후 ~ 테스트벤치 종료까지 측정)
    std::thread power_thread;
    double consumed_uJ = 0.0;
    bool power_ok = set_power_measure(&consumed_uJ, &power_thread);
    if (!power_ok) Log::event("POWER", "power measurement disabled (continue without it)");

    std::thread llm_thread(llmThreadFunc, std::ref(llm));
    std::thread vision_thread(visionThreadFunc);

    vision_thread.join();
    llm_thread.join();

    llm.stop();
    GpuFreq::releaseAll();

    // power end
    if (power_ok) get_power_measure(&consumed_uJ, &power_thread);

    // 최종 요약
    auto s = dvfsController.snapshot();
    auto c = GpuFreq::totalCounters();
    char buf[400];
    std::snprintf(buf, sizeof(buf),
        "LLM requests=%d  llm_time avg=%.1fms min=%.1fms max=%.1fms  queue_wait avg=%.1fms  dropped=%zu",
        llm_stats.count,
        llm_stats.count ? llm_stats.sum_ms / llm_stats.count : 0.0,
        llm_stats.count ? llm_stats.min_ms : 0.0, llm_stats.max_ms,
        llm_stats.count ? llm_stats.sum_wait_ms / llm_stats.count : 0.0,
        vision_ring.dropped());
    Log::event("SUMMARY", buf);
    if (g_mode == DvfsMode::PREDICTIVE) {
        std::snprintf(buf, sizeof(buf),
            "mode=%s  freq up=%d (early %d) down=%d (false_alarm %d)  FSM transitions=%d  final state=%s",
            modeToString(g_mode), c.up, c.early_up, c.down, c.false_alarm_down,
            s.transitions, stateToString(s.state));
    } else {
        std::snprintf(buf, sizeof(buf), "mode=%s  freq up=%d down=%d",
                      modeToString(g_mode), c.up, c.down);
    }
    Log::event("SUMMARY", buf);
    if (power_ok) {
        std::snprintf(buf, sizeof(buf), "energy=%.3f J (%.0f uJ)", consumed_uJ / 1e6, consumed_uJ);
        Log::event("SUMMARY", buf);
    }
    Log::event("SYSTEM", "System Terminated.");
    Log::close();
    return 0;
}
