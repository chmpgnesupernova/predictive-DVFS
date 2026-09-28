#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <atomic>
#include <string>
#include <vector>
#include <map>
#include <cstdio>
#include <cstdlib>
#include <csignal>

#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "vision.h"
#include "dvfs_class.h"
#include "ring_buffer.h"

#define TARGET_FPS 20
#define TIME_SLOT_SEC 60
#define TIME_SLOT_NUM 10
#define RING_BUFFER_SIZE 8

// LLM 자식 프로세스 설정
static const char* MOTION_PLAN_BIN = "./models/motion-plan";
static const char* LLM_MODEL_PATH  = "./models/Llama-3.2-1B-Instruct-Q4_K_M.gguf";

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

// global
PredictiveDVFS dvfsController;
RingBuffer<std::string, RING_BUFFER_SIZE> vision_ring;

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

        // exec 인자는 fork 전에 준비 (fork 이후 child 에서는 exec 만 수행)
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

        // parent
        close(in_pipe[0]);
        close(out_pipe[1]);
        to_child_   = fdopen(in_pipe[1], "w");
        from_child_ = fdopen(out_pipe[0], "r");
        if (!to_child_ || !from_child_) {
            perror("[LLM] fdopen");
            return false;
        }

        // 모델 로드 완료 대기
        std::cout << "[LLM] Waiting for motion-plan to load model...\n";
        return readUntil(LLM_READY_MARKER, nullptr);
    }

    // 요청 1건 전송 후 응답이 끝날 때까지 block
    bool request(const std::string& input, std::string& response) {
        if (fprintf(to_child_, "%s\n", input.c_str()) < 0 || fflush(to_child_) != 0) {
            return false;
        }
        return readUntil(LLM_END_MARKER, &response);
    }

    void stop() {
        if (to_child_)   { fclose(to_child_);   to_child_ = nullptr; }   // EOF -> child 종료
        if (from_child_) {
            char buf[256];
            while (fgets(buf, sizeof(buf), from_child_)) {}               // 잔여 출력 drain
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
    bool readUntil(const std::string& marker, std::string* out) {
        char* line = nullptr;
        size_t cap = 0;
        ssize_t n;
        while ((n = getline(&line, &cap, from_child_)) != -1) {
            std::string s(line, n);
            while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();

            if (s == marker) {
                free(line);
                return true;
            }
            if (out) { *out += s; *out += '\n'; }
            if (!s.empty()) std::cout << "[LLM] " << s << "\n";
        }
        free(line);
        return false;   // child 종료 (EOF)
    }

    pid_t pid_ = -1;
    FILE* to_child_ = nullptr;
    FILE* from_child_ = nullptr;
};

// Robot Middleware
#define REQUEST_INTERVAL_MS 1000
static std::chrono::steady_clock::time_point interval_start;
static bool interval_active = false;   // 첫 HIT 이전에는 interval 없음

Decision robotMiddleware(const std::string& vision_output) {
    std::cout << "[Middleware] Received Vision Output: '" << vision_output << "'\n";
    auto now = std::chrono::steady_clock::now();

    // interval(1s) 진행 중: HOLD (miss 로 세지 않음)
    if (interval_active) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - interval_start).count();
        if (elapsed < REQUEST_INTERVAL_MS) {
            std::cout << "[Middleware] In request interval (" << elapsed << " ms). HOLD.\n";
            return Decision::HOLD;
        }
        interval_active = false;   // interval 종료, 이후 프레임부터 HIT/MISS 판정
    }

    if (vision_output == "apple") {
        std::cout << "[Middleware] Target Object detected! Decided to request LLM.\n";
        interval_start = now;
        interval_active = true;
        return Decision::HIT;
    }

    std::cout << "[Middleware] Non-target object. No LLM request. MISS.\n";
    return Decision::MISS;
}

// LLM Thread
void llmThreadFunc(LlmProcess& llm) {
    std::string current_vision_result;

    // close() 후 버퍼가 비면 pop 이 false 를 반환하며 종료
    while (vision_ring.pop(current_vision_result)) {
        std::cout << "[LLM Thread] Starting LLM Inference for: " << current_vision_result << "...\n";

        auto t0 = std::chrono::steady_clock::now();
        std::string response;
        bool ok = llm.request(current_vision_result, response);
        auto llm_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::steady_clock::now() - t0).count();

        // 요청 1건 종료: 남은 요청이 없으면 GPU LOW 로 down-scaling
        GpuFreq::release();

        if (!ok) {
            std::cerr << "[LLM Thread] motion-plan process terminated unexpectedly.\n";
            vision_ring.close();
            GpuFreq::releaseAll();
            break;
        }

        std::cout << "[LLM Thread] LLM Output Completed for " << current_vision_result
                  << " (" << llm_ms << " ms)\n";
    }
}

// Vision Thread
void visionThreadFunc() {
    const int frame_duration_ms = 1000 / TARGET_FPS; // 50ms
    const int total_frame = TIME_SLOT_NUM * TIME_SLOT_SEC * TARGET_FPS;

    // 1. ONNX Runtime 초기화 및 세션 로드 (루프 외부에서 1회만 실행)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisionInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(6); // Jetson CPU 코어 최적화

    // 모델 경로와 입출력 노드 이름 설정
    const std::string model_path = "./models/mobilenetv2.onnx";
    const char* input_node_name = "input";
    const char* output_node_name = "output";

    Ort::Session session(env, model_path.c_str(), session_options);

    for (int frame_count = 0; frame_count < total_frame; ++frame_count) {
        auto start_time = std::chrono::steady_clock::now();
        std::cout << "\n=== Processing Frame " << (frame_count + 1) << " ===\n";

        // slot 의 이미지를 선택
        int cur_slot = frame_count / (TIME_SLOT_SEC * TARGET_FPS); // slot 당 1200 frame
        std::string img = image_slot[cur_slot];

        // Early Scaling
        dvfsController.scaleFrequencyEarly();

        InferenceResult result = runMobileNetInference(
            img, session, input_node_name, output_node_name
        );

        std::string vision_output;
        if (result.class_id == -1) {
            vision_output = "none"; // 이미지 로드 실패 등 예외 처리
        } else {
            // ID를 문자열로 매핑. 맵에 없는 경우 "object_아이디" 형태로 변환
            if (class_map.find(result.class_id) != class_map.end()) {
                vision_output = class_map[result.class_id];
            } else {
                vision_output = "object_" + std::to_string(result.class_id);
            }
            std::cout << "[Vision Thread] Inference Latency: " << result.inference_time_ms << " ms\n";
            std::cout << "[Vision Thread] Detected: " << vision_output << " (Score: " << result.confidence << ")\n";
        }

        // Middleware 판단 (HIT / MISS / HOLD)
        Decision decision = robotMiddleware(vision_output);

        // Reactive Scaling & State 업데이트
        dvfsController.scaleFrequencyReactive(decision);

        // Ring buffer 를 이용한 LLM 데이터 전달
        if (decision == Decision::HIT) {
            PushResult r = vision_ring.push(vision_output);
            if (r == PushResult::Accepted) {
                GpuFreq::acquire();   // 처리 완료 전까지 GPU down-scaling 방지
            } else if (r == PushResult::Overwrote) {
                std::cout << "[Vision Thread] Ring buffer full, oldest request dropped.\n";
            }
        }

        // 20FPS 유지를 위한 보정
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed < frame_duration_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_duration_ms - elapsed));
        }
    }

    vision_ring.close();
}

int main() {
    std::cout << "Starting VLA Predictive-DVFS Framework...\n";

    // 자식 프로세스가 죽은 뒤 pipe 에 쓰면 SIGPIPE 로 전체가 종료되는 것을 방지
    signal(SIGPIPE, SIG_IGN);

    if (!GpuFreq::init()) {
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

    std::thread llm_thread(llmThreadFunc, std::ref(llm));
    std::thread vision_thread(visionThreadFunc);

    vision_thread.join();
    llm_thread.join();

    llm.stop();
    GpuFreq::releaseAll();

    std::cout << "Dropped LLM requests (ring buffer overwrite): " << vision_ring.dropped() << "\n";
    std::cout << "System Terminated.\n";
    return 0;
}
