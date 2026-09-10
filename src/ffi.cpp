// C ABI 实现：paged-serving 与 tiny-llm 的执行后端桥接。
//
// 策略 2（连续 KV）：block_tables 忽略；序列位置由引擎内部跟踪
// （KV 写入位置以 slot.current_len 为准，与调用方传入 position 解耦）。
// 逐序列执行（step 的第 s 项对应 seq_id = s），多序列语义正确。
// 正常 greedy 请求把末端 RMSNorm / LM head / argmax 组成 device batch，并在 step
// 末尾一次性回传；Transformer layer 的计算融合仍是后续里程碑。

#include "tiny_llm/ffi.h"
#include "elementwise.cuh"
#include "rmsnorm.cuh"
#include "rope.cuh"
#include "sampling.cuh"
#include "tiny_llm/cuda_utils.h"
#include "tiny_llm/execution_common.h"
#include "tiny_llm/gguf_parser.h"
#include "tiny_llm/inference_engine.h" // sampleGreedy
#include "tiny_llm/kv_cache.h"
#include "tiny_llm/model_loader.h"
#include "tiny_llm/transformer.h"
#include "w8a16_matmul.cuh"

#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

constexpr int TLLM_OK = 0;
constexpr int TLLM_ERR = -1;

struct SeqState {
    int  position = 0; // 已处理的 token 数（下一 decode 的绝对位置）
    bool allocated = false;
};

struct TinyLlmHandleImpl {
    tiny_llm::ModelConfig                                    config;
    tiny_llm::ModelWeights                                   weights;
    std::vector<std::unique_ptr<tiny_llm::TransformerLayer>> layers;
    std::unique_ptr<tiny_llm::KVCacheManager>                kv_cache;
    tiny_llm::LayerWorkspace                                 workspace;
    float                                                   *rope_cos = nullptr;
    float                                                   *rope_sin = nullptr;
    half                        *hidden_buf = nullptr; // [max_seq_len * hidden]
    half                        *logits_buf = nullptr; // [vocab_size]
    cudaStream_t                 stream = 0;
    int                          max_batch_size = 1;
    tiny_llm::DeviceBuffer<int>  d_tokens;   // token ids 上传缓冲（gather 需 device 指针）
    tiny_llm::DeviceBuffer<int>  decode_len; // 任务 3.1：decode 可见 KV 长度 (device int)
    tiny_llm::DeviceBuffer<int>  rope_pos;   // 任务 3.2：RoPE 起始位置 (device int)
    tiny_llm::DeviceBuffer<int>  d_next_tokens;  // normal greedy 的逐序列 device 输出
    tiny_llm::DeviceBuffer<half> d_batch_hidden; // [batch, hidden] 的末层状态
    tiny_llm::DeviceBuffer<half> d_batch_logits; // [batch, vocab] 的末端 logits
    std::unordered_map<int, SeqState> sequences;

    // ── 分页 KV（策略 1，max_num_blocks > 0）──
    bool  paged_kv = false;
    int   max_num_blocks = 0;
    int   max_visible_tokens = 0;
    int   block_size = 0;         // paged 模式下保存的分页块大小
    half *paged_k_pool = nullptr; // [L * max_num_blocks * block_size * kv_dim]
    half *paged_v_pool = nullptr;
    half *paged_k_scratch = nullptr; // [max_visible_tokens * kv_dim]
    half *paged_v_scratch = nullptr;
    tiny_llm::DeviceBuffer<int> d_block_tables; // 容量 max_num_blocks
};

// ── 内部工具 ─────────────────────────────────────────────

void set_err(char *err_buf, int err_buf_len, const std::string &msg) {
    if (err_buf && err_buf_len > 0) {
        std::strncpy(err_buf, msg.c_str(), static_cast<size_t>(err_buf_len) - 1);
        err_buf[err_buf_len - 1] = '\0';
    }
}

void embed(TinyLlmHandleImpl *h, const int *tokens, int num_tokens, half *output) {
    // gather_embeddings 的 tokens 是 device 指针：先上传主机 token ids
    if (h->d_tokens.size() < static_cast<size_t>(num_tokens)) {
        h->d_tokens = tiny_llm::DeviceBuffer<int>(static_cast<size_t>(num_tokens));
    }
    h->d_tokens.copyFromHost(tokens, static_cast<size_t>(num_tokens), h->stream);
    tiny_llm::kernels::gather_embeddings(h->d_tokens.data(), h->weights.token_embedding, output,
                                         num_tokens, h->config.hidden_dim, h->config.vocab_size,
                                         h->stream);
}

/// final norm + lm_head + greedy 采样，返回下一 token id。
int sample_from_hidden(TinyLlmHandleImpl *h, half *hidden) {
    // 任务 4.3：final norm + lm_head 走共享 helper（与 InferenceEngine 一致）
    tiny_llm::finalNormAndComputeLogits(hidden, h->weights, h->config, h->logits_buf, h->stream);
    CUDA_CHECK(cudaStreamSynchronize(h->stream));

    std::vector<half> h_logits(static_cast<size_t>(h->config.vocab_size));
    CUDA_CHECK(cudaMemcpy(h_logits.data(), h->logits_buf,
                          static_cast<size_t>(h->config.vocab_size) * sizeof(half),
                          cudaMemcpyDeviceToHost));
    return tiny_llm::InferenceEngine::sampleGreedy(h_logits.data(), h->config.vocab_size);
}

// 每个序列 layer forward 完成后立即保存末层 hidden，避免同一 shared hidden_buf 被
// 下一序列覆盖；同一 stream 保证后续 batch postprocess 读取到完整结果。
void queue_hidden_for_batch(TinyLlmHandleImpl *h, half *hidden, int batch_index) {
    const size_t offset = static_cast<size_t>(batch_index) * h->config.hidden_dim;
    CUDA_CHECK(cudaMemcpyAsync(h->d_batch_hidden.data() + offset, hidden,
                               static_cast<size_t>(h->config.hidden_dim) * sizeof(half),
                               cudaMemcpyDeviceToDevice, h->stream));
}

/// 对最后一步采样的 logits 计算 top-k (token_id, logprob)，写入交错数组。
/// out 布局：s * logprobs_k * 2 起，每候选 (token_id, logprob)。
void compute_logprobs(TinyLlmHandleImpl *h, int seq_idx, int logprobs_k, float *out) {
    if (logprobs_k <= 0 || out == nullptr) return;

    std::vector<half> h_logits(static_cast<size_t>(h->config.vocab_size));
    CUDA_CHECK(cudaMemcpy(h_logits.data(), h->logits_buf,
                          static_cast<size_t>(h->config.vocab_size) * sizeof(half),
                          cudaMemcpyDeviceToHost));

    // softmax（数值稳定）
    float max_l = -1e30f;
    for (const auto &v : h_logits)
        max_l = std::max(max_l, __half2float(v));
    double             sum_exp = 0.0;
    std::vector<float> probs(static_cast<size_t>(h->config.vocab_size));
    for (size_t i = 0; i < h_logits.size(); ++i) {
        probs[i] = std::exp(__half2float(h_logits[i]) - max_l);
        sum_exp += probs[i];
    }
    for (auto &p : probs)
        p = static_cast<float>(p / sum_exp);

    // top-k 选择（插入排序，k 通常很小）
    const int                          k = logprobs_k;
    std::vector<std::pair<int, float>> top;
    for (int v = 0; v < h->config.vocab_size; ++v) {
        if (probs[v] <= 0.0f) continue;
        std::pair<int, float> cand{v, probs[v]};
        bool                  inserted = false;
        for (size_t i = 0; i < top.size(); ++i) {
            if (cand.second > top[i].second) {
                top.insert(top.begin() + static_cast<long>(i), cand);
                inserted = true;
                break;
            }
        }
        if (!inserted && top.size() < static_cast<size_t>(k)) top.push_back(cand);
        if (top.size() > static_cast<size_t>(k)) top.pop_back();
    }

    float *dst = out + static_cast<size_t>(seq_idx) * k * 2;
    for (int i = 0; i < k; ++i) {
        if (i < static_cast<int>(top.size())) {
            dst[i * 2] = static_cast<float>(top[i].first);
            dst[i * 2 + 1] = std::log(top[i].second);
        } else {
            dst[i * 2] = -1.0f;
            dst[i * 2 + 1] = 0.0f;
        }
    }
}

} // namespace

// ── C ABI 入口 ───────────────────────────────────────────

extern "C" {

TinyLlmHandle *tinyllm_load(const char *model_path, const TinyLlmConfig *config, char *err_buf,
                            int err_buf_len) {
    if (model_path == nullptr) {
        set_err(err_buf, err_buf_len, "tinyllm_load: null model_path");
        return nullptr;
    }
    try {
        auto h = std::make_unique<TinyLlmHandleImpl>();

        // 解析 GGUF 提取真实模型配置
        tiny_llm::GGUFParser parser(model_path);
        auto                 parse_result = parser.parse();
        if (parse_result.isErr()) {
            set_err(err_buf, err_buf_len, "GGUF parse: " + parse_result.error());
            return nullptr;
        }
        auto cfg_result = parser.extractModelConfig();
        if (cfg_result.isErr()) {
            set_err(err_buf, err_buf_len, "config: " + cfg_result.error());
            return nullptr;
        }
        h->config = cfg_result.value();

        // 加载权重（GGUF -> W8A16/FP16 上传 GPU）
        auto w_result = tiny_llm::ModelLoader::loadGGUF(model_path, h->config);
        if (w_result.isErr()) {
            set_err(err_buf, err_buf_len, "load weights: " + w_result.error());
            return nullptr;
        }
        h->weights = std::move(w_result.value());

        // KV 生命周期（多序列，max_batch_size 由调用方配置）。
        int max_batch = config ? std::max(1, config->max_batch_size) : 1;
        h->max_batch_size = max_batch;

        const bool use_paged = (config != nullptr && config->max_num_blocks > 0);
        if (use_paged) {
            // 策略 1：分页 KV 池。校验 + 分配 K/V pool 与 scratch；跳过
            // KVCacheManager（h->kv_cache 保持 nullptr），块表由调用方传入。
            if (config->block_size <= 0 || config->max_num_blocks <= 0) {
                set_err(err_buf, err_buf_len,
                        "tinyllm_load: paged KV requires block_size > 0 and max_num_blocks > 0");
                return nullptr;
            }
            h->paged_kv = true;
            h->max_num_blocks = config->max_num_blocks;
            h->max_visible_tokens = config->max_num_blocks * config->block_size;
            h->block_size = config->block_size;
            const int    kv_dim = h->config.num_kv_heads * h->config.head_dim;
            const size_t pool_elems =
                static_cast<size_t>(h->config.num_layers) * static_cast<size_t>(h->max_num_blocks) *
                static_cast<size_t>(config->block_size) * static_cast<size_t>(kv_dim);
            const size_t scratch_elems =
                static_cast<size_t>(h->max_visible_tokens) * static_cast<size_t>(kv_dim);
            CUDA_CHECK(cudaMalloc(&h->paged_k_pool, pool_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_v_pool, pool_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_k_scratch, scratch_elems * sizeof(half)));
            CUDA_CHECK(cudaMalloc(&h->paged_v_scratch, scratch_elems * sizeof(half)));
            h->d_block_tables = tiny_llm::DeviceBuffer<int>(static_cast<size_t>(h->max_num_blocks));
        } else {
            // 策略 2：连续 KV（KVCacheManager）
            tiny_llm::KVCacheConfig kv_config;
            kv_config.num_layers = h->config.num_layers;
            kv_config.num_kv_heads = h->config.num_kv_heads;
            kv_config.head_dim = h->config.head_dim;
            kv_config.max_seq_len = h->config.max_seq_len;
            kv_config.max_batch_size = max_batch;
            auto kv_result = tiny_llm::KVCacheManager::create(kv_config);
            if (kv_result.isErr()) {
                set_err(err_buf, err_buf_len, "KV cache: " + kv_result.error());
                return nullptr;
            }
            h->kv_cache = std::move(kv_result.value());
        }

        // 共享激活工作区 + 层
        h->workspace.allocate(h->config);
        h->layers.reserve(static_cast<size_t>(h->config.num_layers));
        for (int i = 0; i < h->config.num_layers; ++i) {
            h->layers.push_back(std::make_unique<tiny_llm::TransformerLayer>(
                i, h->weights.layers[i], h->config, &h->workspace));
        }

        // 任务 3.1/3.2：decode 可见长度与 RoPE 位置（device int，graph 前置）
        h->decode_len = tiny_llm::DeviceBuffer<int>(1);
        h->rope_pos = tiny_llm::DeviceBuffer<int>(1);
        // 修复（跨流数据竞争）：先创建工作流，RoPE 预计算在其上 launch 并同步。
        // 此前预计算在 cudaStreamCreate 之前执行（h->stream 尚为默认流 0），
        // 而末尾 sync 的是新流，等于从未等待预计算完成——首次推理可能读到
        // 未写完的 cos/sin 表。
        CUDA_CHECK(cudaStreamCreate(&h->stream));

        // RoPE cache + hidden/logits 缓冲
        int half_d = h->config.head_dim / 2;
        CUDA_CHECK(cudaMalloc(&h->rope_cos,
                              static_cast<size_t>(h->config.max_seq_len) * half_d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&h->rope_sin,
                              static_cast<size_t>(h->config.max_seq_len) * half_d * sizeof(float)));
        tiny_llm::kernels::rope_precompute_cache(h->rope_cos, h->rope_sin, h->config.max_seq_len,
                                                 h->config.head_dim, h->config.rope_theta,
                                                 h->stream);
        CUDA_CHECK(cudaMalloc(&h->hidden_buf, static_cast<size_t>(h->config.max_seq_len) *
                                                  h->config.hidden_dim * sizeof(half)));
        CUDA_CHECK(
            cudaMalloc(&h->logits_buf, static_cast<size_t>(h->config.vocab_size) * sizeof(half)));
        CUDA_CHECK(cudaStreamSynchronize(h->stream));

        return reinterpret_cast<TinyLlmHandle *>(h.release());
    } catch (const std::exception &e) {
        set_err(err_buf, err_buf_len, std::string("tinyllm_load: ") + e.what());
        return nullptr;
    }
}

int tinyllm_allocate_sequence(TinyLlmHandle *handle, int seq_id, int num_tokens) {
    if (handle == nullptr || num_tokens <= 0) return TLLM_ERR;
    auto *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    try {
        if (h->paged_kv) {
            // 策略 1：分页 KV 由调用方管理块表；这里只登记序列。
            // 修复：重复分配同一 seq_id 会覆盖登记并泄漏旧序列状态，显式拒绝。
            if (h->sequences.count(seq_id) != 0) {
                if (std::getenv("TLLM_FFI_DEBUG")) {
                    fprintf(stderr, "  [ffi] allocate_sequence(%d) already allocated\n", seq_id);
                }
                return TLLM_ERR;
            }
            SeqState st;
            st.allocated = true;
            h->sequences[seq_id] = st;
            if (std::getenv("TLLM_FFI_DEBUG")) {
                fprintf(stderr, "  [ffi] paged allocate_sequence(%d, %d) ok\n", seq_id, num_tokens);
            }
            return TLLM_OK;
        }
        auto alloc = h->kv_cache->allocateSequence(seq_id, num_tokens);
        if (alloc.isErr()) {
            if (std::getenv("TLLM_FFI_DEBUG")) {
                fprintf(stderr, "  [ffi] allocate_sequence(%d,%d) fail: %s\n", seq_id, num_tokens,
                        alloc.error().c_str());
            }
            return TLLM_ERR;
        }
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] allocate_sequence(%d, %d) ok\n", seq_id, num_tokens);
        }
        SeqState st;
        st.allocated = true;
        h->sequences[seq_id] = st;
        return TLLM_OK;
    } catch (const std::exception &e) {
        if (std::getenv("TLLM_FFI_DEBUG")) {
            fprintf(stderr, "  [ffi] allocate_sequence(%d) threw: %s\n", seq_id, e.what());
        }
        return TLLM_ERR;
    }
}

int tinyllm_free_sequence(TinyLlmHandle *handle, int seq_id) {
    if (handle == nullptr) return TLLM_ERR;
    auto *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    try {
        if (h->paged_kv) {
            // 策略 1：无连续 KV 可释放，仅移除序列登记。
            h->sequences.erase(seq_id);
            return TLLM_OK;
        }
        auto rel = h->kv_cache->releaseSequence(seq_id);
        h->sequences.erase(seq_id);
        return rel.isErr() ? TLLM_ERR : TLLM_OK;
    } catch (...) {
        return TLLM_ERR;
    }
}

int tinyllm_step(TinyLlmHandle *handle, const int *seq_ids, const int *input_tokens,
                 const int *positions, const int *seq_lens, const int *block_tables,
                 const int *num_blocks, const unsigned char *is_prefill, int num_sequences,
                 int *next_tokens, float *logprobs, int logprobs_k) {
    if (handle == nullptr || seq_ids == nullptr || input_tokens == nullptr || seq_lens == nullptr ||
        is_prefill == nullptr || next_tokens == nullptr || num_sequences <= 0) {
        return TLLM_ERR;
    }
    auto     *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    const int hidden = h->config.hidden_dim;

    // logprobs_k 的范围与输出缓冲区必须和双源 ABI 契约一致。
    if (logprobs_k < 0 || logprobs_k > h->config.vocab_size ||
        (logprobs_k > 0 && logprobs == nullptr)) {
        return TLLM_ERR;
    }

    const bool batch_device_postprocess = logprobs_k == 0;
    if (batch_device_postprocess) {
        try {
            const size_t batch_size = static_cast<size_t>(num_sequences);
            if (h->d_next_tokens.size() < batch_size) {
                h->d_next_tokens = tiny_llm::DeviceBuffer<int>(batch_size);
            }
            if (h->d_batch_hidden.size() < batch_size * h->config.hidden_dim) {
                h->d_batch_hidden = tiny_llm::DeviceBuffer<half>(batch_size * h->config.hidden_dim);
            }
            if (h->d_batch_logits.size() < batch_size * h->config.vocab_size) {
                h->d_batch_logits = tiny_llm::DeviceBuffer<half>(batch_size * h->config.vocab_size);
            }
        } catch (...) {
            return TLLM_ERR;
        }
    }

    int offset = 0;
    int table_offset = 0; // 扁平化 block_tables 中前面序列块数之和（策略 1）
    for (int s = 0; s < num_sequences; ++s) {
        const int len = seq_lens[s];
        if (len <= 0) return TLLM_ERR;
        const int seq_id = seq_ids[s];
        auto      it = h->sequences.find(seq_id);
        if (it == h->sequences.end() || !it->second.allocated) return TLLM_ERR;
        auto &st = it->second;

        const int *toks = input_tokens + offset;
        offset += len;

        try {
            if (std::getenv("TLLM_FFI_DEBUG") && !h->paged_kv) {
                auto c = h->kv_cache->getCache(seq_id, 0);
                fprintf(stderr, "  [ffi] seq %d hasSeq=%d cache0=%p kv_seq_len=%d\n", seq_id,
                        (int)h->kv_cache->hasSequence(seq_id), (void *)c.first,
                        h->kv_cache->getSeqLen(seq_id));
            }
            if (h->paged_kv) {
                // ── 策略 1：分页 KV ──
                // 前置校验：块表/计数非空，每序列块数合法且足以容纳本步。
                if (block_tables == nullptr || num_blocks == nullptr) return TLLM_ERR;
                const int nb = num_blocks[s];
                if (nb <= 0 || nb > h->max_num_blocks) return TLLM_ERR;
                const int block_size = h->block_size;
                const int cur_pos = is_prefill[s] ? 0 : st.position;
                const int need_blocks = is_prefill[s] ? (len + block_size - 1) / block_size
                                                      : (cur_pos + 1 + block_size - 1) / block_size;
                if (nb < need_blocks) return TLLM_ERR;

                // 块表上传（逐序列，同一 stream 上顺序执行）
                h->d_block_tables.copyFromHost(block_tables + table_offset, static_cast<size_t>(nb),
                                               h->stream);
                table_offset += nb;

                // 构造视图
                tiny_llm::PagedKVCacheView view;
                view.k_pool = h->paged_k_pool;
                view.v_pool = h->paged_v_pool;
                view.block_table = h->d_block_tables.data();
                view.k_scratch = h->paged_k_scratch;
                view.v_scratch = h->paged_v_scratch;
                view.visible_blocks = nb;
                view.block_size = block_size;
                view.max_num_blocks = h->max_num_blocks;
                view.max_visible_tokens = h->max_visible_tokens;

                if (is_prefill[s]) {
                    // 修复：prefill 长度必须 <= max_seq_len，否则 hidden_buf /
                    // RoPE 表越界（hidden_buf 按 max_seq_len 分配）。
                    if (len > h->config.max_seq_len) return TLLM_ERR;
                    // R5: 不支持对同一序列二次 prefill（KV 从位置 0 重写覆盖
                    // 旧上下文、current_len 双倍累加会静默错乱），显式拒绝。
                    if (st.position != 0) return TLLM_ERR;
                    // Prefill：从绝对位置 0 写入整个 prompt
                    view.position = 0;
                    view.decode_len = nullptr;
                    const int zero_pos = 0;
                    h->rope_pos.copyFromHost(&zero_pos, 1, h->stream);
                    embed(h, toks, len, h->hidden_buf);
                    int li = 0;
                    for (auto &layer : h->layers) {
                        auto r = layer->forwardPaged(h->hidden_buf, view, len, h->rope_pos.data(),
                                                     h->rope_cos, h->rope_sin, h->stream);
                        if (r.isErr()) {
                            if (std::getenv("TLLM_FFI_DEBUG")) {
                                fprintf(stderr, "  [ffi] paged layer %d err: %s\n", li,
                                        r.error().c_str());
                            }
                            return TLLM_ERR;
                        }
                        ++li;
                    }
                    if (batch_device_postprocess) {
                        queue_hidden_for_batch(h, h->hidden_buf + (len - 1) * hidden, s);
                    } else {
                        next_tokens[s] = sample_from_hidden(h, h->hidden_buf + (len - 1) * hidden);
                    }
                    st.position = len;
                } else {
                    // Decode：单个新 token（绝对位置 = st.position）
                    const int pos = st.position;
                    // 修复：绝对位置必须 < max_seq_len，否则 hidden_buf 写入
                    // 与 RoPE cos/sin 表读取越界。
                    if (pos >= h->config.max_seq_len) return TLLM_ERR;
                    const int visible = pos + 1;
                    h->decode_len.copyFromHost(&visible, 1, h->stream);
                    h->rope_pos.copyFromHost(&pos, 1, h->stream);
                    embed(h, toks, 1, h->hidden_buf + static_cast<size_t>(pos) * hidden);
                    half *token_state = h->hidden_buf + static_cast<size_t>(pos) * hidden;
                    view.position = pos;
                    view.decode_len = h->decode_len.data();
                    int li = 0;
                    for (auto &layer : h->layers) {
                        auto r = layer->forwardPaged(token_state, view, 1, h->rope_pos.data(),
                                                     h->rope_cos, h->rope_sin, h->stream);
                        if (r.isErr()) {
                            if (std::getenv("TLLM_FFI_DEBUG")) {
                                fprintf(stderr, "  [ffi] paged decode layer %d err: %s\n", li,
                                        r.error().c_str());
                            }
                            return TLLM_ERR;
                        }
                        ++li;
                    }
                    if (batch_device_postprocess) {
                        queue_hidden_for_batch(h, token_state, s);
                    } else {
                        next_tokens[s] = sample_from_hidden(h, token_state);
                    }
                    st.position = pos + 1;
                }
            } else if (is_prefill[s]) {
                // Prefill：处理完整 prompt，输出最后一个 hidden 的下一 token
                // 修复：prefill 长度必须 <= max_seq_len（hidden_buf 越界防护）。
                if (len > h->config.max_seq_len) return TLLM_ERR;
                // R5: 不支持对同一序列二次 prefill（KV 从位置 0 重写覆盖旧
                // 上下文、advanceSeqLen 累加后 decode 错乱），显式拒绝。
                if (st.position != 0) return TLLM_ERR;
                bool dbg = std::getenv("TLLM_FFI_DEBUG") != nullptr;
                embed(h, toks, len, h->hidden_buf);
                if (dbg) {
                    cudaError_t e = cudaGetLastError();
                    fprintf(stderr, "  [ffi] embed: %s\n", cudaGetErrorString(e));
                }
                // 任务 3.2：prefill 起始位置 0、append 写位置 = 当前可见长度
                const int zero_pos = 0;
                h->rope_pos.copyFromHost(&zero_pos, 1, h->stream);
                h->kv_cache->setAppendPos(0, h->stream);
                int li = 0;
                for (auto &layer : h->layers) {
                    auto r = [&]() -> tiny_llm::Result<void> {
                        try {
                            return layer->forwardPrefill(h->hidden_buf, *h->kv_cache, seq_id, len,
                                                         h->rope_pos.data(), h->rope_cos,
                                                         h->rope_sin, h->stream);
                        } catch (const std::exception &e) {
                            if (std::getenv("TLLM_FFI_DEBUG")) {
                                fprintf(stderr, "  [ffi] layer %d threw: %s\n", li, e.what());
                            }
                            return tiny_llm::Result<void>::err(std::string("threw: ") + e.what());
                        }
                    }();
                    if (r.isErr()) {
                        if (std::getenv("TLLM_FFI_DEBUG")) {
                            fprintf(stderr, "  [ffi] layer %d forwardPrefill err: %s\n", li,
                                    r.error().c_str());
                        }
                        return TLLM_ERR;
                    }
                    if (dbg) {
                        cudaError_t e = cudaDeviceSynchronize();
                        fprintf(stderr, "  [ffi] layer %d: %s\n", li, cudaGetErrorString(e));
                        if (e != cudaSuccess) break;
                    }
                    ++li;
                }
                auto adv = h->kv_cache->advanceSeqLen(seq_id, len);
                if (adv.isErr()) return TLLM_ERR;
                if (batch_device_postprocess) {
                    queue_hidden_for_batch(h, h->hidden_buf + (len - 1) * hidden, s);
                } else {
                    next_tokens[s] = sample_from_hidden(h, h->hidden_buf + (len - 1) * hidden);
                }
                st.position = len;
            } else {
                // Decode：单个新 token（绝对位置由引擎跟踪，与 KV 写入一致）
                const int pos = st.position;
                // 修复：绝对位置必须 < max_seq_len（hidden_buf / RoPE 表越界防护）。
                if (pos >= h->config.max_seq_len) return TLLM_ERR;
                embed(h, toks, 1, h->hidden_buf + static_cast<size_t>(pos) * hidden);
                half *token_state = h->hidden_buf + static_cast<size_t>(pos) * hidden;
                // 任务 3.1/3.2：把 appendKV 后的可见长度、RoPE 位置与
                // append 写位置写入 device 缓冲
                const int decode_len = h->kv_cache->getSeqLen(seq_id) + 1;
                h->decode_len.copyFromHost(&decode_len, 1, h->stream);
                h->rope_pos.copyFromHost(&pos, 1, h->stream);
                h->kv_cache->setAppendPos(h->kv_cache->getSeqLen(seq_id), h->stream);
                for (auto &layer : h->layers) {
                    auto r =
                        layer->forward(token_state, *h->kv_cache, seq_id, pos, h->decode_len.data(),
                                       h->rope_pos.data(), h->rope_cos, h->rope_sin, h->stream);
                    if (r.isErr()) return TLLM_ERR;
                }
                auto adv = h->kv_cache->advanceSeqLen(seq_id, 1);
                if (adv.isErr()) return TLLM_ERR;
                if (batch_device_postprocess) {
                    queue_hidden_for_batch(h, token_state, s);
                } else {
                    next_tokens[s] = sample_from_hidden(h, token_state);
                }
                st.position = pos + 1;
            }
        } catch (...) {
            return TLLM_ERR;
        }

        if (logprobs_k > 0 && logprobs != nullptr) {
            compute_logprobs(h, s, logprobs_k, logprobs);
        }
    }

    if (batch_device_postprocess) {
        try {
            tiny_llm::finalNormAndComputeLogitsBatch(h->d_batch_hidden.data(), num_sequences,
                                                     h->weights, h->config,
                                                     h->d_batch_logits.data(), h->stream);
            tiny_llm::kernels::greedy_argmax_batch(h->d_batch_logits.data(), num_sequences,
                                                   h->config.vocab_size, h->d_next_tokens.data(),
                                                   h->stream);
            CUDA_CHECK(cudaMemcpyAsync(next_tokens, h->d_next_tokens.data(),
                                       static_cast<size_t>(num_sequences) * sizeof(int),
                                       cudaMemcpyDeviceToHost, h->stream));
            CUDA_CHECK(cudaStreamSynchronize(h->stream));
        } catch (...) {
            return TLLM_ERR;
        }
    }

    (void)positions;    // 策略 2：位置由引擎内部跟踪（策略 1 不使用 positions）
    (void)block_tables; // 策略 2：连续 KV，忽略分页表（策略 1 已在循环内使用）
    (void)num_blocks; // 策略 2：连续 KV，忽略逐序列块计数（策略 1 已在循环内使用）
    return TLLM_OK;
}

void tinyllm_free(TinyLlmHandle *handle) {
    if (handle == nullptr) return;
    auto *h = reinterpret_cast<TinyLlmHandleImpl *>(handle);
    // LayerWorkspace 与各组件 RAII 释放；显式释放裸指针
    if (h->rope_cos) {
        cudaFree(h->rope_cos);
        h->rope_cos = nullptr;
    }
    if (h->rope_sin) {
        cudaFree(h->rope_sin);
        h->rope_sin = nullptr;
    }
    if (h->hidden_buf) {
        cudaFree(h->hidden_buf);
        h->hidden_buf = nullptr;
    }
    if (h->logits_buf) {
        cudaFree(h->logits_buf);
        h->logits_buf = nullptr;
    }
    // 分页 KV pool / scratch（裸指针，显式释放；d_block_tables 是 RAII）
    if (h->paged_k_pool) {
        cudaFree(h->paged_k_pool);
        h->paged_k_pool = nullptr;
    }
    if (h->paged_v_pool) {
        cudaFree(h->paged_v_pool);
        h->paged_v_pool = nullptr;
    }
    if (h->paged_k_scratch) {
        cudaFree(h->paged_k_scratch);
        h->paged_k_scratch = nullptr;
    }
    if (h->paged_v_scratch) {
        cudaFree(h->paged_v_scratch);
        h->paged_v_scratch = nullptr;
    }
    if (h->stream) {
        cudaStreamDestroy(h->stream);
        h->stream = 0;
    }
    tiny_llm::ModelLoader::freeWeights(h->weights);
    delete h;
}

} // extern "C"
