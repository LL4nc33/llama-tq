// MTP draft bookkeeping on a tiny random qwen35moe model with one nextn (MTP) layer, CPU only.
//
// checks:
//   1. the target exports its post-norm hidden state (llama_get_embeddings_nextn) for every batch token,
//      not only for the output tokens
//   2. after a prompt and several draft/verify/accept rounds (with rejected drafts), the MTP head
//      drafts exactly what a reference built from scratch drafts: every token x_p paired with the
//      target hidden state h_{p-1} at position p, the first draft seeded with h of the last accepted token
//   3. a second request with the same prompt drafts exactly the same as the first one (no state leaks)
//   4. with two sequences in one batch, the nextn rows are in batch order even when the ubatch split reorders the
//      tokens (split_equal takes the same number of tokens from each sequence)
//   5. two sequences sharing one speculative state, with their prompts and verify tokens in the same target batches
//      (mirrored into the draft context once per batch) and drafted in the same draft batches, draft exactly what each
//      sequence drafts alone; for a unified KV cache and for one stream per sequence

#include "common.h"
#include "speculative.h"
#include "llama.h"
#include "llama-cpp.h"
#include "gguf.h"
#include "ggml-cpp.h"

#include "../src/llama-arch.h"
#include "../src/llama-ext.h"
#include "../src/llama-model-saver.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const uint32_t N_VOCAB = 128;
static const uint32_t N_EMBD  = 256;
static const uint32_t N_HEAD  = 2;
static const uint32_t N_FF    = 384;
static const uint32_t N_CTX   = 256;

static void set_tensor_data(struct ggml_tensor * tensor, void * userdata) {
    std::hash<std::string> hasher;
    std::mt19937 gen(hasher(tensor->name) + *(const size_t *) userdata);
    std::normal_distribution<float> dis(0.0f, 1.0e-2f);

    // norm weights near 1 (as in trained models), otherwise every norm shrinks the signal by 100x and the
    // MTP logits hardly depend on the hidden state input
    const float offset = strstr(tensor->name, "norm") != nullptr ? 1.0f : 0.0f;

    const int64_t ne = ggml_nelements(tensor);
    if (tensor->type == GGML_TYPE_F32) {
        std::vector<float> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = offset + dis(gen);
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else if (tensor->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(ne);
        for (int64_t i = 0; i < ne; i++) {
            tmp[i] = ggml_fp32_to_fp16(offset + dis(gen));
        }
        ggml_backend_tensor_set(tensor, tmp.data(), 0, ggml_nbytes(tensor));
    } else {
        GGML_ABORT("unexpected tensor type");
    }
}

// two trunk layers (gated delta net + full attention) and one MTP layer
static gguf_context_ptr make_gguf() {
    gguf_context_ptr ret(gguf_init_empty());
    llama_model_saver ms(LLM_ARCH_QWEN35MOE, ret.get());

    const uint32_t n_layer     = 3;
    const uint32_t n_embd_head = N_EMBD / N_HEAD;

    ms.add_kv(LLM_KV_GENERAL_ARCHITECTURE,          llm_arch_name(LLM_ARCH_QWEN35MOE));
    ms.add_kv(LLM_KV_VOCAB_SIZE,                    N_VOCAB);
    ms.add_kv(LLM_KV_CONTEXT_LENGTH,                N_CTX);
    ms.add_kv(LLM_KV_EMBEDDING_LENGTH,              N_EMBD);
    ms.add_kv(LLM_KV_BLOCK_COUNT,                   n_layer);
    ms.add_kv(LLM_KV_NEXTN_PREDICT_LAYERS,          uint32_t(1));
    ms.add_kv(LLM_KV_FEED_FORWARD_LENGTH,           N_FF);
    ms.add_kv(LLM_KV_FULL_ATTENTION_INTERVAL,       uint32_t(2));
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT,          N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_HEAD_COUNT_KV,       N_HEAD);
    ms.add_kv(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,   1e-5f);
    ms.add_kv(LLM_KV_ROPE_DIMENSION_SECTIONS,       std::vector<uint32_t>({n_embd_head/4, n_embd_head/4, n_embd_head/4, n_embd_head/4}));
    ms.add_kv(LLM_KV_TOKENIZER_MODEL,               "no_vocab");
    ms.add_kv(LLM_KV_EXPERT_FEED_FORWARD_LENGTH,    N_FF);
    ms.add_kv(LLM_KV_EXPERT_SHARED_FEED_FORWARD_LENGTH, N_FF);
    ms.add_kv(LLM_KV_EXPERT_COUNT,                  uint32_t(2));
    ms.add_kv(LLM_KV_EXPERT_USED_COUNT,             uint32_t(1));
    ms.add_kv(LLM_KV_SSM_INNER_SIZE,                uint32_t(256));
    ms.add_kv(LLM_KV_SSM_CONV_KERNEL,               uint32_t(4));
    ms.add_kv(LLM_KV_SSM_STATE_SIZE,                uint32_t(128));
    ms.add_kv(LLM_KV_SSM_TIME_STEP_RANK,            N_HEAD);
    ms.add_kv(LLM_KV_SSM_GROUP_COUNT,               uint32_t(2));

    return ret;
}

static llama_context * make_ctx(llama_model * model, bool mtp, bool embeddings, uint32_t n_rs_seq,
        uint32_t n_seq = 1, bool kv_unified = false) {
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = kv_unified ? N_CTX : N_CTX * n_seq;
    cparams.n_batch         = 64;
    cparams.n_ubatch        = 64;
    cparams.n_seq_max       = n_seq;
    cparams.kv_unified      = kv_unified;
    cparams.n_threads       = 4;
    cparams.n_threads_batch = 4;
    cparams.embeddings      = embeddings;
    cparams.n_rs_seq        = n_rs_seq;
    if (mtp) {
        cparams.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
    }
    return llama_init_from_model(model, cparams);
}

static llama_token argmax(const float * logits) {
    return (llama_token) (std::max_element(logits, logits + N_VOCAB) - logits);
}

// normalized mean squared error = mse(a, b) / mse(a, 0), infinite if a is zero or anything is not finite
static double nmse(const float * a, const float * b, size_t n) {
    double mse_a_b = 0.0;
    double mse_a_0 = 0.0;
    for (size_t i = 0; i < n; i++) {
        mse_a_b += (a[i] - b[i]) * (a[i] - b[i]);
        mse_a_0 += a[i] * a[i];
    }
    const double res = mse_a_b / mse_a_0;
    return std::isfinite(res) ? res : INFINITY;
}

struct request_result {
    llama_tokens tokens;                 // prompt + accepted tokens, without id_last
    llama_token  id_last = 0;            // sampled, not yet decoded
    llama_tokens draft_last;             // the draft of the final round
    std::vector<llama_tokens> drafts;    // all drafts
    std::vector<float> logits_dft_last;  // ctx_dft logits of the last draft step
    bool ok = false;
};

// one request as the server runs it: prompt, then draft / verify / accept rounds
static request_result run_request(common_speculative * spec, llama_context * ctx_tgt, llama_context * ctx_dft,
        const llama_tokens & prompt, const std::vector<int> & n_accept_schedule) {
    request_result res;

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    llama_memory_seq_rm(mem_tgt, 0, 0, -1);

    {
        llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        if (llama_decode(ctx_tgt, batch) != 0) {
            fprintf(stderr, "prompt decode failed\n");
            llama_batch_free(batch);
            return res;
        }
        llama_memory_seq_rm(mem_dft, 0, 0, -1);
        common_speculative_process(spec, batch);
        llama_batch_free(batch);
    }
    common_speculative_begin(spec, 0, prompt);

    res.tokens  = prompt;
    res.id_last = argmax(llama_get_logits_ith(ctx_tgt, -1));

    for (size_t round = 0; round <= n_accept_schedule.size(); ++round) {
        llama_tokens draft;
        {
            auto & dp = common_speculative_get_draft_params(spec, 0);
            dp.drafting = true;
            dp.n_max    = 3;
            dp.n_past   = (llama_pos) res.tokens.size();
            dp.id_last  = res.id_last;
            dp.prompt   = &res.tokens;
            dp.result   = &draft;
            common_speculative_draft(spec);
        }
        res.drafts.push_back(draft);

        if (round == n_accept_schedule.size()) {
            // final round: keep the draft and the logits of the last MTP step for the reference
            res.draft_last = draft;
            const float * logits = llama_get_logits_ith(ctx_dft, -1);
            res.logits_dft_last.assign(logits, logits + N_VOCAB);
            break;
        }

        // verify [id_last, draft...]
        const llama_pos p0 = (llama_pos) res.tokens.size();
        llama_batch batch = llama_batch_init(1 + (int32_t) draft.size(), 0, 1);
        common_batch_add(batch, res.id_last, p0, { 0 }, true);
        for (size_t i = 0; i < draft.size(); ++i) {
            common_batch_add(batch, draft[i], p0 + 1 + (llama_pos) i, { 0 }, true);
        }
        if (llama_decode(ctx_tgt, batch) != 0) {
            fprintf(stderr, "verify decode failed\n");
            llama_batch_free(batch);
            return res;
        }
        llama_memory_seq_rm(mem_dft, 0, p0, -1);
        common_speculative_process(spec, batch);

        // the bookkeeping does not depend on why tokens are accepted: accept per schedule
        const int n_acc = std::min<int>(n_accept_schedule[round], (int) draft.size());
        const llama_token id_next = argmax(llama_get_logits_ith(ctx_tgt, n_acc));
        llama_batch_free(batch);

        common_speculative_accept(spec, 0, (uint16_t) n_acc);

        res.tokens.push_back(res.id_last);
        res.tokens.insert(res.tokens.end(), draft.begin(), draft.begin() + n_acc);
        res.id_last = id_next;

        if (!llama_memory_seq_rm(mem_tgt, 0, (llama_pos) res.tokens.size(), -1)) {
            fprintf(stderr, "target rollback failed\n");
            return res;
        }
        llama_memory_seq_rm(mem_dft, 0, (llama_pos) res.tokens.size(), -1);
    }

    res.ok = true;
    return res;
}

struct seq_request {
    llama_tokens     prompt;
    std::vector<int> schedule;      // number of accepted draft tokens per round
    int              round_start;   // round in which the prompt joins the target batch, -1: never
};

// several requests as the server runs them over one speculative state: per round, one draft call drafts for all
// generating sequences, then one target batch holds their verify tokens followed by the prompts that start in this
// round. the batch is mirrored into the draft context once, after trimming each sequence from its first position
static std::vector<request_result> run_requests(common_speculative * spec, llama_context * ctx_tgt, llama_context * ctx_dft,
        const std::vector<seq_request> & reqs) {
    const int n_req = (int) reqs.size();

    std::vector<request_result> res(n_req);

    auto * mem_tgt = llama_get_memory(ctx_tgt);
    auto * mem_dft = llama_get_memory(ctx_dft);

    llama_memory_clear(mem_tgt, true);
    llama_memory_clear(mem_dft, true);

    std::vector<int>          n_round(n_req, -1); // completed verify rounds, -1: prompt not processed yet
    std::vector<llama_tokens> drafts(n_req);

    for (int round = 0; ; ++round) {
        bool any_drafting = false;
        for (llama_seq_id s = 0; s < n_req; ++s) {
            drafts[s].clear();
            if (n_round[s] < 0 || n_round[s] > (int) reqs[s].schedule.size()) {
                continue;
            }
            auto & dp = common_speculative_get_draft_params(spec, s);
            dp.drafting = true;
            dp.n_max    = 3;
            dp.n_past   = (llama_pos) res[s].tokens.size();
            dp.id_last  = res[s].id_last;
            dp.prompt   = &res[s].tokens;
            dp.result   = &drafts[s];
            any_drafting = true;
        }
        if (any_drafting) {
            common_speculative_draft(spec);
        }

        llama_batch batch = llama_batch_init(64, 0, 1);
        std::vector<int32_t> i_batch(n_req, -1); // batch index of id_last (verify) or of the last prompt token

        for (llama_seq_id s = 0; s < n_req; ++s) {
            if (n_round[s] < 0 || n_round[s] > (int) reqs[s].schedule.size()) {
                continue;
            }
            res[s].drafts.push_back(drafts[s]);
            if (n_round[s] == (int) reqs[s].schedule.size()) {
                // final round: keep the draft, done
                res[s].draft_last = drafts[s];
                res[s].ok = true;
                n_round[s]++;
                continue;
            }
            const llama_pos p0 = (llama_pos) res[s].tokens.size();
            i_batch[s] = batch.n_tokens;
            common_batch_add(batch, res[s].id_last, p0, { s }, true);
            for (size_t i = 0; i < drafts[s].size(); ++i) {
                common_batch_add(batch, drafts[s][i], p0 + 1 + (llama_pos) i, { s }, true);
            }
        }
        for (llama_seq_id s = 0; s < n_req; ++s) {
            if (reqs[s].round_start != round) {
                continue;
            }
            for (size_t i = 0; i < reqs[s].prompt.size(); ++i) {
                common_batch_add(batch, reqs[s].prompt[i], (llama_pos) i, { s }, i + 1 == reqs[s].prompt.size());
            }
            i_batch[s] = batch.n_tokens - 1;
        }

        if (batch.n_tokens == 0) {
            llama_batch_free(batch);
            break;
        }

        if (llama_decode(ctx_tgt, batch) != 0) {
            fprintf(stderr, "target decode failed\n");
            llama_batch_free(batch);
            return res;
        }
        for (llama_seq_id s = 0; s < n_req; ++s) {
            for (int32_t k = 0; k < batch.n_tokens; ++k) {
                if (batch.seq_id[k][0] == s) {
                    llama_memory_seq_rm(mem_dft, s, batch.pos[k], -1);
                    break;
                }
            }
        }
        common_speculative_process(spec, batch);

        for (llama_seq_id s = 0; s < n_req; ++s) {
            if (i_batch[s] < 0) {
                continue;
            }
            if (n_round[s] < 0) {
                // prompt processed
                common_speculative_begin(spec, s, reqs[s].prompt);
                res[s].tokens  = reqs[s].prompt;
                res[s].id_last = argmax(llama_get_logits_ith(ctx_tgt, i_batch[s]));
                n_round[s] = 0;
                continue;
            }

            const int n_acc = std::min<int>(reqs[s].schedule[n_round[s]], (int) drafts[s].size());
            const llama_token id_next = argmax(llama_get_logits_ith(ctx_tgt, i_batch[s] + n_acc));

            common_speculative_accept(spec, s, (uint16_t) n_acc);

            res[s].tokens.push_back(res[s].id_last);
            res[s].tokens.insert(res[s].tokens.end(), drafts[s].begin(), drafts[s].begin() + n_acc);
            res[s].id_last = id_next;

            if (!llama_memory_seq_rm(mem_tgt, s, (llama_pos) res[s].tokens.size(), -1)) {
                fprintf(stderr, "target rollback failed\n");
                llama_batch_free(batch);
                return res;
            }
            llama_memory_seq_rm(mem_dft, s, (llama_pos) res[s].tokens.size(), -1);
            n_round[s]++;
        }

        llama_batch_free(batch);
    }

    return res;
}

int main(void) {
    common_init();
    ggml_backend_load_all();

    llama_model_params mparams = llama_model_default_params();
    mparams.load_mtp = true;
    mparams.devices  = nullptr;
    {
        // CPU only
        static ggml_backend_dev_t devs[2] = { ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU), nullptr };
        mparams.devices = devs;
    }

    size_t seed = 42;
    gguf_context_ptr gguf_ctx = make_gguf();
    llama_model_ptr model(llama_model_init_from_user(gguf_ctx.get(), set_tensor_data, &seed, mparams));
    if (!model) {
        fprintf(stderr, "failed to create the model\n");
        return 1;
    }
    if (!llama_model_has_mtp(model.get())) {
        fprintf(stderr, "the model has no MTP layer\n");
        return 1;
    }

    llama_context_ptr ctx_tgt(make_ctx(model.get(), false, false, 4));
    llama_context_ptr ctx_dft(make_ctx(model.get(), true,  false, 0));
    llama_context_ptr ref_tgt(make_ctx(model.get(), false, true,  0));
    llama_context_ptr ref_dft(make_ctx(model.get(), true,  false, 0));
    if (!ctx_tgt || !ctx_dft || !ref_tgt || !ref_dft) {
        fprintf(stderr, "failed to create the contexts\n");
        return 1;
    }
    llama_set_embeddings_nextn(ref_dft.get(), true, true);

    common_params_speculative sparams;
    sparams.types         = { COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
    sparams.draft.ctx_tgt = ctx_tgt.get();
    sparams.draft.ctx_dft = ctx_dft.get();
    sparams.draft.n_max   = 3;
    sparams.draft.n_min   = 0;
    sparams.draft.p_min   = 0.0f; // always draft n_max tokens
    common_speculative_ptr spec(common_speculative_init(sparams, 1));
    if (!spec) {
        fprintf(stderr, "failed to init speculative decoding\n");
        return 1;
    }

    llama_tokens prompt;
    {
        std::mt19937 gen(1234);
        std::uniform_int_distribution<int> dis(0, N_VOCAB - 1);
        for (int i = 0; i < 24; ++i) {
            prompt.push_back(dis(gen));
        }
    }

    const std::vector<int> schedule = { 0, 3, 1, 0, 2, 3, 0, 1 };

    bool ok = true;

    // request 1
    request_result r1 = run_request(spec.get(), ctx_tgt.get(), ctx_dft.get(), prompt, schedule);
    if (!r1.ok) {
        return 1;
    }

    // reference: the target hidden states of the accepted sequence, decoded in one batch
    const size_t L = r1.tokens.size();
    std::vector<float> H(L * N_EMBD);
    {
        llama_batch batch = llama_batch_init((int32_t) L, 0, 1);
        for (size_t i = 0; i < L; ++i) {
            common_batch_add(batch, r1.tokens[i], (llama_pos) i, { 0 }, true);
        }
        if (llama_decode(ref_tgt.get(), batch) != 0) {
            fprintf(stderr, "reference target decode failed\n");
            return 1;
        }
        llama_batch_free(batch);
        for (size_t i = 0; i < L; ++i) {
            memcpy(H.data() + i * N_EMBD, llama_get_embeddings_ith(ref_tgt.get(), (int32_t) i), N_EMBD * sizeof(float));
        }
    }

    // check 1: the target exported its post-norm hidden state for every prompt token. ctx_tgt's last
    // decode was a verify batch, so re-run the prompt (only the last token is an output)
    {
        llama_memory_seq_rm(llama_get_memory(ctx_tgt.get()), 0, 0, -1);
        llama_batch batch = llama_batch_init((int32_t) prompt.size(), 0, 1);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(batch, prompt[i], (llama_pos) i, { 0 }, i + 1 == prompt.size());
        }
        if (llama_decode(ctx_tgt.get(), batch) != 0) {
            fprintf(stderr, "prompt decode failed\n");
            return 1;
        }
        llama_batch_free(batch);

        double worst = 0.0;
        for (size_t i = 0; i < prompt.size(); ++i) {
            const float * h = llama_get_embeddings_nextn_ith(ctx_tgt.get(), (int32_t) i);
            if (h == nullptr) {
                fprintf(stderr, "check 1: no nextn row for token %zu\n", i);
                return 1;
            }
            worst = std::max(worst, nmse(H.data() + i * N_EMBD, h, N_EMBD));
        }
        const bool pass = worst < 1e-6;
        printf("check 1 (target nextn export for all tokens): worst nmse = %.3e %s\n", worst, pass ? "OK" : "FAIL");
        ok &= pass;
    }

    // check 2: replay the MTP head from scratch: x_p paired with h_{p-1} at position p (h_{-1} = 0),
    // then the three draft steps of the final round
    {
        const int32_t n_b = (int32_t) L + 1;
        llama_batch batch = llama_batch_init(n_b, N_EMBD, 1);
        batch.token = (llama_token *) malloc(sizeof(llama_token) * n_b);

        common_batch_clear(batch);
        for (size_t i = 0; i < L; ++i) {
            common_batch_add(batch, r1.tokens[i], (llama_pos) i, { 0 }, false);
            if (i == 0) {
                memset(batch.embd, 0, N_EMBD * sizeof(float));
            } else {
                memcpy(batch.embd + i * N_EMBD, H.data() + (i - 1) * N_EMBD, N_EMBD * sizeof(float));
            }
        }
        if (llama_decode(ref_dft.get(), batch) != 0) {
            fprintf(stderr, "reference MTP prefill failed\n");
            return 1;
        }

        std::vector<float> h_prev(H.end() - N_EMBD, H.end());
        llama_token tok = r1.id_last;
        std::vector<float> logits_ref;
        bool tokens_match = r1.draft_last.size() == 3;
        for (size_t step = 0; step < 3 && tokens_match; ++step) {
            common_batch_clear(batch);
            common_batch_add(batch, tok, (llama_pos) (L + step), { 0 }, true);
            memcpy(batch.embd, h_prev.data(), N_EMBD * sizeof(float));
            if (llama_decode(ref_dft.get(), batch) != 0) {
                fprintf(stderr, "reference MTP step failed\n");
                return 1;
            }
            const float * logits = llama_get_logits_ith(ref_dft.get(), 0);
            logits_ref.assign(logits, logits + N_VOCAB);

            std::vector<float> sorted(logits_ref);
            std::sort(sorted.begin(), sorted.end(), std::greater<float>());
            const bool clear_winner = sorted[0] - sorted[1] > 1e-3f * std::fabs(sorted[0]);
            if (clear_winner && argmax(logits) != r1.draft_last[step]) {
                fprintf(stderr, "check 2: draft step %zu: got %d, reference %d\n", step, r1.draft_last[step], argmax(logits));
                tokens_match = false;
            }

            const float * h = llama_get_embeddings_nextn_ith(ref_dft.get(), 0);
            h_prev.assign(h, h + N_EMBD);
            tok = r1.draft_last[step];
        }

        free(batch.token);
        batch.token = nullptr;
        llama_batch_free(batch);

        // the final draft step's input is (draft[1], h of step 1): its logits are what ctx_dft holds
        const double err = tokens_match ? nmse(logits_ref.data(), r1.logits_dft_last.data(), N_VOCAB) : 1.0;
        const bool pass = tokens_match && err < 1e-4;
        printf("check 2 (MTP drafts match a replay of h_{p-1} + x_p pairs): nmse = %.3e %s\n", err, pass ? "OK" : "FAIL");
        ok &= pass;
    }

    // check 3: the same request again drafts exactly the same
    {
        request_result r2 = run_request(spec.get(), ctx_tgt.get(), ctx_dft.get(), prompt, schedule);
        if (!r2.ok) {
            return 1;
        }
        const bool pass = r2.drafts == r1.drafts && r2.tokens == r1.tokens && r2.logits_dft_last == r1.logits_dft_last;
        printf("check 3 (second identical request drafts the same): %s\n", pass ? "OK" : "FAIL");
        ok &= pass;
    }

    // two sequences: seq 0 with a longer prompt than seq 1, so that a batch holding both prompts is split unevenly
    llama_tokens prompt1;
    {
        std::mt19937 gen(5678);
        std::uniform_int_distribution<int> dis(0, N_VOCAB - 1);
        for (int i = 0; i < 9; ++i) {
            prompt1.push_back(dis(gen));
        }
    }
    const std::vector<seq_request> reqs = {
        { prompt,  schedule,                0 },
        { prompt1, { 2, 0, 3, 1, 1, 3, 0 }, 0 },
    };

    for (const bool kv_unified : { false, true }) {
        const char * kv_name = kv_unified ? "unified KV" : "KV stream per sequence";

        llama_context_ptr ctx2_tgt(make_ctx(model.get(), false, false, 4, 2, kv_unified));
        llama_context_ptr ctx2_dft(make_ctx(model.get(), true,  false, 0, 2, kv_unified));
        if (!ctx2_tgt || !ctx2_dft) {
            fprintf(stderr, "failed to create the two-sequence contexts\n");
            return 1;
        }

        // also enables the nextn export of ctx2_tgt
        common_params_speculative sparams2 = sparams;
        sparams2.draft.ctx_tgt = ctx2_tgt.get();
        sparams2.draft.ctx_dft = ctx2_dft.get();
        common_speculative_ptr spec2(common_speculative_init(sparams2, 2));
        if (!spec2) {
            fprintf(stderr, "failed to init two-sequence speculative decoding\n");
            return 1;
        }

        // check 4: decode both prompts in one batch, compare the nextn rows with each prompt decoded alone
        {
            std::vector<std::vector<float>> H_ref(reqs.size());
            for (size_t s = 0; s < reqs.size(); ++s) {
                const auto & p = reqs[s].prompt;
                llama_memory_clear(llama_get_memory(ref_tgt.get()), true);
                llama_batch batch = llama_batch_init((int32_t) p.size(), 0, 1);
                for (size_t i = 0; i < p.size(); ++i) {
                    common_batch_add(batch, p[i], (llama_pos) i, { 0 }, true);
                }
                if (llama_decode(ref_tgt.get(), batch) != 0) {
                    fprintf(stderr, "reference target decode failed\n");
                    return 1;
                }
                llama_batch_free(batch);
                H_ref[s].resize(p.size() * N_EMBD);
                for (size_t i = 0; i < p.size(); ++i) {
                    memcpy(H_ref[s].data() + i * N_EMBD, llama_get_embeddings_ith(ref_tgt.get(), (int32_t) i), N_EMBD * sizeof(float));
                }
            }

            llama_memory_clear(llama_get_memory(ctx2_tgt.get()), true);
            llama_batch batch = llama_batch_init(64, 0, 1);
            for (llama_seq_id s = 0; s < (llama_seq_id) reqs.size(); ++s) {
                const auto & p = reqs[s].prompt;
                for (size_t i = 0; i < p.size(); ++i) {
                    common_batch_add(batch, p[i], (llama_pos) i, { s }, i + 1 == p.size());
                }
            }
            if (llama_decode(ctx2_tgt.get(), batch) != 0) {
                fprintf(stderr, "two-prompt decode failed\n");
                return 1;
            }
            llama_batch_free(batch);

            double worst = 0.0;
            int32_t k = 0;
            for (size_t s = 0; s < reqs.size(); ++s) {
                for (size_t i = 0; i < reqs[s].prompt.size(); ++i, ++k) {
                    const float * h = llama_get_embeddings_nextn_ith(ctx2_tgt.get(), k);
                    if (h == nullptr) {
                        fprintf(stderr, "check 4: no nextn row for batch token %d\n", k);
                        return 1;
                    }
                    worst = std::max(worst, nmse(H_ref[s].data() + i * N_EMBD, h, N_EMBD));
                }
            }
            const bool pass = worst < 1e-6;
            printf("check 4 (nextn rows of a two-sequence batch in batch order, %s): worst nmse = %.3e %s\n",
                    kv_name, worst, pass ? "OK" : "FAIL");
            ok &= pass;
        }

        // check 5: both sequences together (prompts in the same batch, and seq 1 joining while seq 0 generates)
        // draft the same as each sequence alone
        {
            std::vector<request_result> alone(reqs.size());
            for (size_t s = 0; s < reqs.size(); ++s) {
                std::vector<seq_request> reqs_alone = reqs;
                for (size_t o = 0; o < reqs.size(); ++o) {
                    reqs_alone[o].round_start = o == s ? 0 : -1;
                }
                alone[s] = run_requests(spec2.get(), ctx2_tgt.get(), ctx2_dft.get(), reqs_alone)[s];
                if (!alone[s].ok) {
                    fprintf(stderr, "check 5: sequence %zu alone did not finish\n", s);
                    return 1;
                }
            }

            for (const int round_start_1 : { 0, 2 }) {
                std::vector<seq_request> reqs_tog = reqs;
                reqs_tog[1].round_start = round_start_1;
                const auto tog = run_requests(spec2.get(), ctx2_tgt.get(), ctx2_dft.get(), reqs_tog);

                bool pass = true;
                for (size_t s = 0; s < reqs.size(); ++s) {
                    pass &= tog[s].ok && tog[s].drafts == alone[s].drafts && tog[s].tokens == alone[s].tokens;
                }
                printf("check 5 (two sequences drafted together match each alone, %s, seq 1 joins in round %d): %s\n",
                        kv_name, round_start_1, pass ? "OK" : "FAIL");
                ok &= pass;
            }
        }
    }

    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
