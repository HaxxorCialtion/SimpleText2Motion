// test_hidden.cpp — verify hidden-layer extraction for Qwen3
#include "llama.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <cstdlib>

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <path-to-gguf> [layer_index=6] [prompt]\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const int    layer_idx  = (argc >= 3) ? atoi(argv[2]) : 6;
    const char * prompt     = (argc >= 4) ? argv[3] : "The quick brown fox jumps over the lazy dog.";

    llama_backend_init();

    // load model
    auto mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999; // offload all to GPU
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }

    const int n_embd = llama_model_n_embd(model);
    printf("model loaded: n_embd=%d\n", n_embd);

    // create context with hidden layer extraction enabled
    auto cparams = llama_context_default_params();
    cparams.n_ctx = 2048;
    cparams.n_batch = 2048;
    cparams.hidden_layer_output = layer_idx;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "ctx create failed\n"); return 1; }

    // tokenize
    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(256);
    int n = llama_tokenize(vocab, prompt, (int) strlen(prompt),
                           toks.data(), (int) toks.size(), true, false);
    if (n < 0) { fprintf(stderr, "tokenize failed\n"); return 1; }
    toks.resize(n);
    printf("tokenized: %d tokens\n", n);

    // decode (prefill)
    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "decode failed\n"); return 1;
    }

    // fetch hidden
    const float * h = llama_get_hidden_layer(ctx);
    int32_t       nt = llama_hidden_layer_n_tokens(ctx);

    printf("\n== hidden layer %d ==\n", layer_idx);
    printf("ptr=%p n_tokens=%d n_embd=%d total floats=%zu (%.2f KB)\n",
           (const void*)h, nt, n_embd, (size_t)nt * n_embd,
           (size_t)nt * n_embd * sizeof(float) / 1024.0);

    if (!h || nt <= 0) {
        fprintf(stderr, "ERROR: hidden is null or empty!\n");
        return 2;
    }

    // dump first 8 floats of first/middle/last token
    auto dump = [&](int i, const char * tag) {
        const float * row = h + (size_t) i * n_embd;
        printf("%s token[%d] [0..7]: % .4f % .4f % .4f % .4f % .4f % .4f % .4f % .4f\n",
               tag, i, row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7]);
    };
    dump(0,       "first ");
    dump(nt / 2,  "middle");
    dump(nt - 1,  "last  ");

    // sanity: are any rows all zeros / NaN?
    int n_zero = 0, n_nan = 0;
    for (int i = 0; i < nt; i++) {
        const float * row = h + (size_t) i * n_embd;
        float s = 0;
        bool has_nan = false;
        for (int k = 0; k < n_embd; k++) {
            if (row[k] != row[k]) has_nan = true;
            s += row[k] * row[k];
        }
        if (has_nan) n_nan++;
        if (s < 1e-12f) n_zero++;
    }
    printf("\nsanity: %d rows all-zero, %d rows with NaN (out of %d)\n", n_zero, n_nan, nt);
    if (n_zero > 0 || n_nan > 0) {
        fprintf(stderr, "WARNING: suspicious values found\n");
    }

    // second decode: add one more token, check we get (1, n_embd)
    printf("\n== second decode (single token) ==\n");
    llama_token next = toks.back();
    llama_batch b2 = llama_batch_get_one(&next, 1);
    if (llama_decode(ctx, b2) != 0) {
        fprintf(stderr, "decode #2 failed\n"); return 1;
    }
    int32_t nt2 = llama_hidden_layer_n_tokens(ctx);
    const float * h2 = llama_get_hidden_layer(ctx);
    printf("n_tokens=%d (should be 1)\n", nt2);
    if (h2) dump(0, "new   ");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("\nOK\n");
    return 0;
}
