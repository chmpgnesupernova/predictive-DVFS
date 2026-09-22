#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <array>
#include <numeric>

// OpenCV & ONNX Runtime Headers
#include <opencv4/opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

// 추론 결과 구조체
struct InferenceResult {
    int class_id;
    float confidence;
    double inference_time_ms;
};

// --- MobileNet ONNX CPU 추론 함수 ---
InferenceResult runMobileNetInference(
    const std::string& image_path,
    Ort::Session& session,
    const char* input_name,
    const char* output_name) {

    // 1. OpenCV 이미지 로드
    cv::Mat img = cv::imread(image_path);
    if (img.empty()) {
        std::cerr << "[Error] Failed to load image: " << image_path << std::endl;
        return {-1, 0.0f, 0.0};
    }

    // 2. Preprocessing (MobileNet Standard: 224x224, BGR -> RGB)
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(224, 224));
    cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);

    // ImageNet Mean & Std Dev
    const std::array<float, 3> mean = {0.485f, 0.456f, 0.406f};
    const std::array<float, 3> std_dev = {0.229f, 0.224f, 0.225f};

    // HWC -> CHW 변환 및 정규화 ((Pixel/255.0 - Mean) / Std)
    std::vector<float> input_tensor_values(1 * 3 * 224 * 224);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 224; ++h) {
            for (int w = 0; w < 224; ++w) {
                float pixel = resized_img.at<cv::Vec3b>(h, w)[c] / 255.0f;
                float normalized_pixel = (pixel - mean[c]) / std_dev[c];
                input_tensor_values[c * 224 * 224 + h * 224 + w] = normalized_pixel;
            }
        }
    }

    // 3. ONNX Input Tensor 매핑 (1, 3, 224, 224)
    std::vector<int64_t> input_shape = {1, 3, 224, 224};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        input_tensor_values.data(), 
        input_tensor_values.size(), 
        input_shape.data(), 
        input_shape.size()
    );

    // 4. ONNX Inference 실행 및 Latency 측정
    const char* input_names[] = {input_name};
    const char* output_names[] = {output_name};

    auto start_time = std::chrono::high_resolution_clock::now();

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1, 
        output_names, 1
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // 5. Postprocessing (Top-1 Class ID 및 Score 추출)
    float* float_arr = output_tensors[0].GetTensorMutableData<float>();
    auto tensor_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t num_classes = tensor_info.GetElementCount(); // ImageNet 기준 보통 1000

    int max_class_id = 0;
    float max_score = float_arr[0];

    for (size_t i = 1; i < num_classes; ++i) {
        if (float_arr[i] > max_score) {
            max_score = float_arr[i];
            max_class_id = static_cast<int>(i);
        }
    }

    return {max_class_id, max_score, elapsed_ms};
}

int main(int argc, char** argv) {
    const std::string model_path = "mobilenetv2.onnx";
    
    // 테스트할 이미지 파일 목록
    std::vector<std::string> test_images = {
        "apple.jpg", "orange.jpg", "test3.jpg", "test4.jpg"
    };

    // 1. ONNX Runtime 세션 설정 (Jetson Orin Nano CPU 최적화)
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "MobileNet_CPU_Test");
    Ort::SessionOptions session_options;

    // Jetson Orin Nano CPU 코어 수(6개) 활용
    session_options.SetIntraOpNumThreads(6);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "[System] Loading ONNX model: " << model_path << " on CPU..." << std::endl;
    
    Ort::Session session(env, model_path.c_str(), session_options);

    // 모델의 입력/출력 노드 이름 (일반적인 PyTorch export 기준)
    // netron.app 등으로 본인 모델의 노드 이름을 확인 후 다를 경우 수정하세요.
    const char* input_node_name = "input";
    const char* output_node_name = "output";

    std::cout << "\n=======================================================" << std::endl;
    std::cout << " Starting Vision Model Inference Benchmark (CPU)" << std::endl;
    std::cout << "=======================================================\n" << std::endl;

    double total_latency = 0.0;
    int valid_frames = 0;

    for (size_t i = 0; i < test_images.size(); ++i) {
        InferenceResult result = runMobileNetInference(
            test_images[i], session, input_node_name, output_node_name
        );

        if (result.class_id != -1) {
            std::cout << "Frame [" << (i + 1) << "/" << test_images.size() << "] "
                      << "Image: " << test_images[i] << "\n"
                      << "  -> Class ID : " << result.class_id << "\n"
                      << "  -> Score    : " << result.confidence << "\n"
                      << "  -> Latency  : " << result.inference_time_ms << " ms\n"
                      << "-------------------------------------------------------" << std::endl;

            // Warm-up(첫 프레임)을 제외한 평균 Latency 계산
            if (i > 0) {
                total_latency += result.inference_time_ms;
                valid_frames++;
            }
        }
    }

    if (valid_frames > 0) {
        std::cout << "\n[Summary] Average Latency (excl. warmup): " 
                  << (total_latency / valid_frames) << " ms" << std::endl;
    }

    return 0;
}
