#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <string>
#include <vector>
#include <atomic>
#include <cstdlib>

// ==========================================
// 1. Predictive-DVFS Controller (4-State FSM)
// ==========================================
enum class PredictorState {
    STRONGLY_TAKEN,
    WEAKLY_TAKEN,
    WEAKLY_NOT_TAKEN,
    STRONGLY_NOT_TAKEN
};

class PredictiveDVFSController {
private:
    PredictorState current_state = PredictorState::STRONGLY_TAKEN; // 초기 상태
    std::mutex state_mutex;
    bool is_scaled_high = false;

    // GPU Frequency Scaling 모사 함수 (실제 환경에서는 NVML API 또는 Driver sysfs 제어)
    void scale_gpu_frequency(bool high_freq) {
        // DVFS Frequency Scaling Latency (~15ms) 모사
        std::this_thread::sleep_for(std::chrono::milliseconds(15));
        is_scaled_high = high_freq;
        std::cout << "  [GPU Kernel] Frequency Scaled -> " << (high_freq ? "HIGH (100% Clock)" : "LOW (Power Save)") 
                  << " (Latency 15ms overhead absorbed)\n";
    }

public:
    // 현재 상태 기반 예측 (Taken / Not Taken)
    bool predict_taken() {
        std::lock_guard<std::mutex> lock(state_mutex);
        return (current_state == PredictorState::STRONGLY_TAKEN || 
                current_state == PredictorState::WEAKLY_TAKEN);
    }

    // Vision 추론 시작 시 Early Scaling 시도
    bool trigger_early_scaling() {
        bool will_scale = predict_taken();
        if (will_scale) {
            std::cout << "  [DVFS Controller] Prediction: TAKEN -> Early Frequency Scaling Triggered!\n";
            scale_gpu_frequency(true);
        } else {
            std::cout << "  [DVFS Controller] Prediction: NOT TAKEN -> Maintain Low Frequency.\n";
            is_scaled_high = false;
        }
        return will_scale;
    }

    // Prediction Miss(Not Taken 예측 후 실제 LLM 요청 발생) 시 Late Scaling 적용
    void ensure_high_frequency_late() {
        if (!is_scaled_high) {
            std::cout << "  [DVFS Controller] Miss Occurred! Executing Late Frequency Scaling (TTFT Penalty ~15ms)...\n";
            scale_gpu_frequency(true);
        }
    }

    // 상태 천이도(State Transition Diagram) 기반 업데이트
    void update_state(bool actual_llm_request) {
        std::lock_guard<std::mutex> lock(state_mutex);
        PredictorState old_state = current_state;

        switch (current_state) {
            case PredictorState::STRONGLY_TAKEN:
                if (!actual_llm_request) current_state = PredictorState::WEAKLY_TAKEN; // Miss
                break;
            case PredictorState::WEAKLY_TAKEN:
                if (actual_llm_request) current_state = PredictorState::STRONGLY_TAKEN; // Hit
                else current_state = PredictorState::WEAKLY_NOT_TAKEN; // Miss
                break;
            case PredictorState::WEAKLY_NOT_TAKEN:
                if (actual_llm_request) current_state = PredictorState::WEAKLY_TAKEN; // Hit
                else current_state = PredictorState::STRONGLY_NOT_TAKEN; // Miss
                break;
            case PredictorState::STRONGLY_NOT_TAKEN:
                if (actual_llm_request) current_state = PredictorState::WEAKLY_NOT_TAKEN; // Hit
                break;
        }

        std::cout << "  [DVFS Controller] State Updated: " << state_to_string(old_state) 
                  << " -> " << state_to_string(current_state) 
                  << " (Actual Request: " << (actual_llm_request ? "Hit" : "Miss") << ")\n";
    }

private:
    std::string state_to_string(PredictorState s) {
        switch (s) {
            case PredictorState::STRONGLY_TAKEN: return "Strongly Taken";
            case PredictorState::WEAKLY_TAKEN: return "Weakly Taken";
            case PredictorState::WEAKLY_NOT_TAKEN: return "Weakly Not Taken";
            case PredictorState::STRONGLY_NOT_TAKEN: return "Strongly Not Taken";
        }
        return "";
    }
};

// ==========================================
// 2. Robot Middleware (Simple Object Recognition Filter)
// ==========================================
class RobotMiddleware {
public:
    // 감지된 객체에 따라 LLM 동작 제어 필요 여부 결정
    static bool decide_llm_request(const std::string& detected_object) {
        // 예: 복잡한 대처가 필요한 객체("obstacle", "unknown_tool")인 경우 LLM 요청
        if (detected_object == "obstacle" || detected_object == "unknown_tool") {
            return true;
        }
        return false; // 단순 환경("empty", "clear_path")은 Vision으로만 처리
    }
};

// ==========================================
// 3. LLM Worker Thread
// ==========================================
class LLMWorker {
private:
    std::queue<std::string> request_queue;
    std::mutex queue_mutex;
    std::condition_variable cv;
    bool stop_flag = false;
    std::thread worker_thread;

    void run() {
        while (true) {
            std::string prompt;
            {
                std::unique_lock<std::mutex> lock(queue_mutex);
                cv.wait(lock, [this]() { return !request_queue.empty() || stop_flag; });

                if (stop_flag && request_queue.empty()) break;

                prompt = request_queue.front();
                request_queue.pop();
            }

            std::cout << "\n  [LLM Thread] Executing llama-cli with prompt: \"" << prompt << "\"\n";
            
            // system()을 통해 llama-cli 바이너리 호출
            std::string command = "./llama-cli -m models/llama-3.2-1b.gguf -p \"" + prompt + "\" -n 16 > /dev/null 2>&1";
            int ret = std::system(command.c_str());
            (void)ret; // Unused variable 경고 방지

            std::cout << "  [LLM Thread] LLM Motion Planning Completed.\n\n";
        }
    }

public:
    LLMWorker() {
        worker_thread = std::thread(&LLMWorker::run, this);
    }

    ~LLMWorker() {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            stop_flag = true;
        }
        cv.notify_one();
        if (worker_thread.joinable()) {
            worker_thread.join();
        }
    }

    void push_request(const std::string& prompt) {
        {
            std::lock_guard<std::mutex> lock(queue_mutex);
            request_queue.push(prompt);
        }
        cv.notify_one();
    }
};

// ==========================================
// 4. Vision Model Thread (20 FPS Simulation)
// ==========================================
void vision_thread_func(PredictiveDVFSController& dvfs, LLMWorker& llm_worker, std::atomic<bool>& running) {
    // 20 FPS = 프레임당 50ms 주기
    const std::chrono::milliseconds frame_duration(50);
    
    // 시뮬레이션용 데이터셋 (20 FPS 영상 프레임 대체)
    std::vector<std::string> frame_stream = {
        "obstacle",     // Hit -> Predict Taken maintained
        "obstacle",     // Hit
        "clear_path",   // Miss -> Weakly Taken으로 전이
        "clear_path",   // Miss -> Weakly Not Taken으로 전이
        "clear_path",   // Miss -> Strongly Not Taken으로 전이
        "unknown_tool", // Hit -> Late Scaling 발생! (Weakly Not Taken으로 전이)
        "obstacle"      // Hit
    };

    size_t frame_idx = 0;

    while (running && frame_idx < frame_stream.size()) {
        auto start_time = std::chrono::steady_clock::now();
        std::string current_frame_object = frame_stream[frame_idx++];

        std::cout << "\n================ [Frame " << frame_idx << "] ================\n";

        // Step 1: Vision 추론 시작 직후 Predictive DVFS 발동
        bool early_scaled = dvfs.trigger_early_scaling();

        // Step 2: Vision Model (YOLOv8) 추론 수행 (약 50ms 소요)
        // -> Early Scaling Latency(15ms)가 이 Vision Inference 시간 뒤로 은닉됨 (Sequence Chart 왼쪽 참조)
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        std::cout << "  [Vision Model (YOLOv8)] Detected Object: " << current_frame_object << "\n";

        // Step 3: Middleware에서 LLM 호출 여부 판단
        bool llm_needed = RobotMiddleware::decide_llm_request(current_frame_object);
        std::cout << "  [Robot Middleware] Decided LLM Request: " << (llm_needed ? "YES" : "NO") << "\n";

        // Step 4: Late Scaling 처리 및 LLM 호출 dispatch
        if (llm_needed) {
            if (!early_scaled) {
                // 예측 실패(Not Taken으로 예측했으나 실제로 필요했던 경우) -> Late Scaling 지연 발생
                dvfs.ensure_high_frequency_late();
            }
            llm_worker.push_request("Navigate around " + current_frame_object);
        }

        // Step 5: Stateful Predictor 업데이트 (Hit / Miss 판정)
        dvfs.update_state(llm_needed);

        // Step 6: 20 FPS 주기 맞춤 (50ms)
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time);
        if (elapsed < frame_duration) {
            std::this_thread::sleep_for(frame_duration - elapsed);
        }
    }
    running = false;
}

// ==========================================
// 5. Main Execution
// ==========================================
int main() {
    std::cout << "Starting Predictive-DVFS VLA Controller Simulation...\n";

    PredictiveDVFSController dvfs_controller;
    LLMWorker llm_worker;
    std::atomic<bool> running{true};

    // Vision 모델 스레드 실행
    std::thread vision_thread(vision_thread_func, std::ref(dvfs_controller), std::ref(llm_worker), std::ref(running));

    if (vision_thread.joinable()) {
        vision_thread.join();
    }

    std::cout << "\nSimulation Finished Successfully.\n";
    return 0;
}
