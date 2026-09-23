#include <onnxruntime_cxx_api.h>
#include <thread>
#include <string>

// vision thread routine
bool vision_thread_routine (){

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "Benchmark");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(4);
    Ort::Session session(env, "model.onnx", session_options);

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_float_array.data(), input_float_array.size(), input_shape.data(), input_shape.size());

    // vision inference
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, 
                                      input_names.data(), &input_tensor, 1, 
                                      output_names.data(), 1); // output string to 
    return true;
}

// llm thread routine
bool llm_thread_routine(std::string vision_output){
    std::string prompt = vision_output;
    std::system ("./stateful-early-scaling --prompt %s -m Llama-3.2-1B.gguf", prompt);
    return true;
}

// middle-ware, suppose we have middleware. 
bool decide_llm_request (std::string vision_output){
    bool res = false;
    std::string object = vision_output.json().object; // extract "object" feild from output
    
    switch (object){
        case tennis_ball:
            res = true;
        case apple:
            res = true;
        default:
            break;
    }

    if (res) {
        std::async (stateful_early_scaling_controller, res); // signal to early scaling controller, to return the real result 
    }    

    msleep (5); // supposed to have middleware, so set the latency 5ms
}

// stateful-early-scaling-controller
class stateful_early_scaling_controller {
private:
    /*
     enum {
        STRONGLY_NOT_TAKEN 0,
        WEAKLY_NOT_TAKEN 1,
        WEAKLY_NOT_TAKEN 2,
        STRONGLY_NOT_TAKEN 3
     }
     */
    int state; 
    int prev_state;

public:
    bool switch_state (){
        // when middle-ware gave signal, compute next state with state , prev 
    }
    

}

int main (){

    while (true) {

    }
}
