// motion-plan.cpp
// llama.cpp examples/simple-chat 기반.
// main.cpp 의 LlmProcess 가 자식 프로세스로 1회 실행하여 상주시키고,
// ring buffer 에서 pop 한 vision 결과를 stdin 으로 한 줄씩 보낸다.
//
// 프로토콜 (stdout 은 parent 와의 통신 채널이므로 로그는 stderr 로):
//   시작 시        : "<<MOTION_PLAN_READY>>\n"
//   요청 1건 (stdin): "<vision_output>\n"
//   응답            : 생성 텍스트 ... "\n<<MOTION_PLAN_END>>\n"
//   stdin EOF / 빈 줄 → 종료

#include "llama.h"
#include <clocale>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

static const char * READY_MARKER = "<<MOTION_PLAN_READY>>";
static const char * END_MARKER   = "<<MOTION_PLAN_END>>";

static const char * SYSTEM_PROMPT =
    "You are a motion planner for a robot arm. "
    "Given the name of a detected object, output a short, numbered list of "
    "primitive actions (e.g. move_to, grasp, lift, place) to pick it up.";

static void print_usage(int, char ** argv) {
    fprintf(stderr, "\nexample usage:\n");
    fprintf(stderr, "\n    %s -m model.gguf [-c context_size] [-ngl n_gpu_layers] [-n n_predict]\n", argv[0]);
    fprintf(stderr, "\n");
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path;
    int ngl       = 99;
    int n_ctx     = 2048;
    int n_predict = 256;   // 응답 길이 상한

    // parse command line arguments
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) { model_path = argv[++i]; }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-c") == 0) {
                if (i + 1 < argc) { n_ctx = std::stoi(argv[++i]); }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-ngl") == 0) {
                if (i + 1 < argc) { ngl = std::stoi(argv[++i]); }
                else { print_usage(argc, argv); return 1; }
            } else if (strcmp(argv[i], "-n") == 0) {
                if (i + 1 < argc) { n_predict = std::stoi(argv[++i]); }
                else { print_usage(argc, argv); return 1; }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = ngl;

    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        llama_model_free(model);
        return 1;
    }

    // initialize the sampler
    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    // helper function to evaluate a prompt and generate a response
    auto generate = [&](const std::string & prompt) {
        std::string response;

        const bool is_first = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) == -1;

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        int n_generated = 0;
        while (n_generated < n_predict) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx_cur  = llama_n_ctx(ctx);
            int n_ctx_used = llama_memory_seq_pos_max(llama_get_memory(ctx), 0) + 1;
            if (n_ctx_used + batch.n_tokens > n_ctx_cur) {
                fprintf(stderr, "context size exceeded\n");
                break;   // exit(0) 대신 break: 상주 프로세스는 다음 요청을 계속 받아야 함
            }

            int ret = llama_decode(ctx, batch);
            if (ret != 0) {
                GGML_ABORT("failed to decode, ret = %d\n", ret);
            }

            // sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;
            ++n_generated;

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };

    const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);
    std::vector<char> formatted(llama_n_ctx(ctx));

    // 모델 로드 완료를 parent 에 알림
    printf("%s\n", READY_MARKER);
    fflush(stdout);

    // ---- simple-chat 의 std::getline(std::cin, user) 루프 ----
    // 입력은 parent(main.cpp) 가 ring buffer 에서 pop 하여 stdin 으로 보낸 vision 결과
    std::string vision_output;
    while (std::getline(std::cin, vision_output)) {
        if (vision_output.empty()) {
            break;
        }

        // 요청 간 독립: 이전 대화(KV cache) 와 sampler 상태 초기화
        llama_memory_clear(llama_get_memory(ctx), true);
        llama_sampler_reset(smpl);

        std::string user = "Detected object: " + vision_output + ". Generate the motion plan.";
        llama_chat_message messages[] = {
            {"system", SYSTEM_PROMPT},
            {"user",   user.c_str()},
        };
        const size_t n_msg = sizeof(messages) / sizeof(messages[0]);

        int new_len = llama_chat_apply_template(tmpl, messages, n_msg, true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages, n_msg, true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            printf("\n%s\n", END_MARKER);   // parent 가 block 되지 않도록 마커는 항상 출력
            fflush(stdout);
            continue;
        }

        std::string prompt(formatted.begin(), formatted.begin() + new_len);

        // generate a response
        generate(prompt);

        // 응답 종료 마커
        printf("\n%s\n", END_MARKER);
        fflush(stdout);
    }

    // free resources
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    return 0;
}
