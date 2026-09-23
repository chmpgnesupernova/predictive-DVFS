#include <iostream>
#include <thread>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <string>
#include <queue>
#include <map>

#include "vision_inference.h"
#include "predictive_dvfs_class.h"

// global
PredictiveDVFS dvfsController;
std::queue<std::string> llm_task_queue;
std::mutex llm_mtx;
std::condition_variable llm_cv;
std::atomic<bool> system_running(true);

// Robot Middleware 
bool robotMiddleware(const std::string& vision_output) {
    std::cout << "[Middleware] Received Vision Output: '" << vision_output << "'\n";

    if (vision_output == "apple" || vision_output == "orange") {
        std::cout << "[Middleware] Target Object detected! Decided to request LLM.\n";
        return true;
    }
    
    std::cout << "[Middleware] Non-target object. No LLM request.\n";
    return false;
}

// LLM Thread 
void llmThreadFunc() {
    while (system_running) {
        std::string current_vision_result;

        {
            std::unique_lock<std::mutex> lock(llm_mtx);
            llm_cv.wait(lock, [] { return !llm_task_queue.empty() || !system_running; });
            
            if (!system_running && llm_task_queue.empty()) break;

            current_vision_result = llm_task_queue.front();
            llm_task_queue.pop();
        }

        std::cout << "[LLM Thread] Starting LLM Inference for: " << current_vision_result << "...\n";
        std::string cmd_start_llm = "sudo ./stateful-early-scaling -m ./Llama3.2" + current_vision_result + "' > /dev/null";
        int ret1 = system(cmd_start_llm.c_str());
        (void)ret1;
        
        std::string cmd_down_freq = "sudo echo userspace > /sys/class/devfreq/170000.gpu/governor \
                                     sudo echo 306000000 > /sys/calss/devfreq/170000.gpu/userspace/set_freq";
        int ret2 = system(cmd_down_freq.c_str());
        (void)ret2;
        
        std::cout << "[LLM Thread] LLM Output Completed for " << current_vision_result << ".\n";
    }
}

// Vision Thread
void visionThreadFunc() {
    const int target_fps = 20;
    const int frame_duration_ms = 1000 / target_fps; // 50ms
    
    // 1. ONNX Runtime 초기화 및 세션 로드 (루프 외부에서 1회만 실행)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VisionInference");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(6); // Jetson CPU 코어 최적화
    
    // 모델 경로와 입출력 노드 이름 설정
    const std::string model_path = "mobilenetv2.onnx";
    const char* input_node_name = "input";
    const char* output_node_name = "output";
    
    Ort::Session session(env, model_path.c_str(), session_options);

    // 테스트용 프레임 이미지 리스트 
    std::vector<std::string> frame_images = {
        "test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg"
    };

    // (옵션) ImageNet Class ID를 문자열로 매핑하기 위한 간단한 딕셔너리
    // 실제 ImageNet 기준: 사과(948), 오렌지(950), 바나나(954)
    std::map<int, std::string> class_map = {
        {948, "apple"},
        {950, "orange"},
        {954, "banana"}
    };

    for (size_t frame_count = 0; frame_count < frame_images.size(); ++frame_count) {
        auto start_time = std::chrono::steady_clock::now();
        std::cout << "\n=== Processing Frame " << (frame_count + 1) << " ===\n";

        // Early Scaling
        dvfsController.scaleFrequencyEarly();

        /*
         * runMobileNetInference 함수 사용하여 vision_output 생성
         */
        std::cout << "[Vision Thread] Running inference on " << frame_images[frame_count] << "...\n";
        
        InferenceResult result = runMobileNetInference(
            frame_images[frame_count], session, input_node_name, output_node_name
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

        // Middleware 판단
        bool decide_llm_request = robotMiddleware(vision_output);

        // Reactive Scaling & State 업데이트
        dvfsController.scaleFrequencyReactive(decide_llm_request);

        // Queue를 이용한 LLM 데이터 전달
        if (decide_llm_request) {
            {
                std::lock_guard<std::mutex> lock(llm_mtx);
                llm_task_queue.push(vision_output);
            }
            llm_cv.notify_one();
        }

        // 20FPS 유지를 위한 보정
        auto end_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        if (elapsed < frame_duration_ms) {
            std::this_thread::sleep_for(std::chrono::milliseconds(frame_duration_ms - elapsed));
        }
    }

    system_running = false;
    llm_cv.notify_all();
}

int main() {
    std::cout << "Starting VLA Predictive-DVFS Framework...\n";

    std::thread llm_thread(llmThreadFunc);
    std::thread vision_thread(visionThreadFunc);

    vision_thread.join();
    llm_thread.join();

    std::cout << "System Terminated.\n";
    return 0;
}
