#include "tiny_llm/kv_cache.h"
#include "elementwise.cuh"
#include "tiny_llm/logger.h"
#include "tiny_llm/validator.h"
#include <algorithm>

namespace tiny_llm {

Result<std::unique_ptr<KVCacheManager>> KVCacheManager::create(const KVCacheConfig &config) {
    // Validate configuration to prevent overflow
    if (config.num_layers <= 0 || config.num_kv_heads <= 0 || config.head_dim <= 0 ||
        config.max_seq_len <= 0 || config.max_batch_size <= 0) {
        return Result<std::unique_ptr<KVCacheManager>>::err(
            "KVCacheManager: invalid configuration parameters");
    }

    // Create instance using private constructor
    auto manager = std::unique_ptr<KVCacheManager>(new KVCacheManager(config));

    // Calculate per-slot memory size with overflow check
    size_t kv_per_layer =
        static_cast<size_t>(config.num_kv_heads) * config.max_seq_len * config.head_dim;

    size_t kv_total = kv_per_layer * static_cast<size_t>(config.num_layers) * 2;

    if (kv_total > SIZE_MAX / sizeof(half)) {
        return Result<std::unique_ptr<KVCacheManager>>::err("KVCacheManager: memory size overflow");
    }
    manager->slot_size_ = kv_total * sizeof(half);

    // Total pool size for all batch slots
    if (manager->slot_size_ > SIZE_MAX / static_cast<size_t>(config.max_batch_size)) {
        return Result<std::unique_ptr<KVCacheManager>>::err("KVCacheManager: pool size overflow");
    }
    manager->pool_size_ = manager->slot_size_ * static_cast<size_t>(config.max_batch_size);

    // Allocate GPU memory pool
    cudaError_t err = cudaMalloc(&manager->memory_pool_, manager->pool_size_);
    if (err != cudaSuccess) {
        return Result<std::unique_ptr<KVCacheManager>>::err(
            std::string("KVCacheManager: cudaMalloc failed: ") + cudaGetErrorString(err));
    }
    err = cudaMemset(manager->memory_pool_, 0, manager->pool_size_);
    if (err != cudaSuccess) {
        cudaFree(manager->memory_pool_);
        manager->memory_pool_ = nullptr;
        return Result<std::unique_ptr<KVCacheManager>>::err(
            std::string("KVCacheManager: cudaMemset failed: ") + cudaGetErrorString(err));
    }

    // Initialize slots
    manager->slots_.resize(config.max_batch_size);
    for (auto &slot : manager->slots_) {
        slot.active = false;
        slot.seq_id = -1;
        slot.current_len = 0;
        slot.max_len = config.max_seq_len;
    }

    // 任务 3.2：append 写位置的 device int 缓冲（CUDA Graph 重放前置）。
    // 必须显式清零（与 memory_pool_ 同样的约定）：cudaMalloc 不保证返回清零内存，
    // 调用方若未先 setAppendPos，appendKV 会按垃圾值写 K/V（静默写错位置）。
    manager->append_pos_ = DeviceBuffer<int>(1);
    err = cudaMemset(manager->append_pos_.data(), 0, sizeof(int));
    if (err != cudaSuccess) {
        return Result<std::unique_ptr<KVCacheManager>>::err(
            std::string("KVCacheManager: append_pos_ init failed: ") + cudaGetErrorString(err));
    }

    return Result<std::unique_ptr<KVCacheManager>>::ok(std::move(manager));
}

KVCacheManager::~KVCacheManager() {
    if (memory_pool_) {
        cudaFree(memory_pool_);
        memory_pool_ = nullptr;
    }
}

size_t KVCacheManager::calculateOffset(int slot_idx, int layer_idx, bool is_value) const {
    // Memory layout per slot:
    // [Layer 0 K][Layer 0 V][Layer 1 K][Layer 1 V]...
    size_t kv_per_layer =
        static_cast<size_t>(config_.num_kv_heads) * config_.max_seq_len * config_.head_dim;

    size_t slot_offset = slot_idx * slot_size_ / sizeof(half);
    size_t layer_offset = layer_idx * kv_per_layer * 2;
    size_t kv_offset = is_value ? kv_per_layer : 0;

    return slot_offset + layer_offset + kv_offset;
}

int KVCacheManager::findFreeSlot() const {
    for (int i = 0; i < static_cast<int>(slots_.size()); ++i) {
        if (!slots_[i].active) {
            return i;
        }
    }
    return -1;
}

Result<int> KVCacheManager::allocateSequence(int max_len) {
    return allocateSequence(next_seq_id_, max_len);
}

Result<int> KVCacheManager::allocateSequence(int seq_id, int max_len) {
    // 修复：重复分配同一 seq_id 会覆盖 seq_to_slot_ 映射，旧 slot 永久泄漏
    // （active=true 但失去引用，无法 release）。显式拒绝重复分配。
    if (seq_to_slot_.find(seq_id) != seq_to_slot_.end()) {
        return Result<int>::err("Sequence already allocated: " + std::to_string(seq_id));
    }

    // Validate max_len
    if (max_len <= 0 || max_len > config_.max_seq_len) {
        return Result<int>::err("Invalid max_len: " + std::to_string(max_len) + " (must be 1-" +
                                std::to_string(config_.max_seq_len) + ")");
    }

    // Find free slot
    int slot_idx = findFreeSlot();
    if (slot_idx < 0) {
        return Result<int>::err("KV cache exhausted: no free slots available. "
                                "Used: " +
                                std::to_string(getUsedMemory()) +
                                " bytes, "
                                "Total: " +
                                std::to_string(getTotalMemory()) + " bytes");
    }

    // Allocate slot（显式 seq_id 供 C ABI 与调用方 id 对齐）
    slots_[slot_idx].seq_id = seq_id;
    slots_[slot_idx].current_len = 0;
    slots_[slot_idx].max_len = max_len;
    slots_[slot_idx].active = true;

    // Zero the whole slot so a re-used slot never exposes stale KV data from a
    // previously released sequence (the slot size is derived from the pool's
    // max_seq_len, not the per-allocation max_len).
    size_t      slot_offset_bytes = static_cast<size_t>(slot_idx) * slot_size_;
    cudaError_t zero_err = cudaMemset(
        reinterpret_cast<unsigned char *>(memory_pool_) + slot_offset_bytes, 0, slot_size_);
    if (zero_err != cudaSuccess) {
        slots_[slot_idx].active = false;
        slots_[slot_idx].seq_id = -1;
        return Result<int>::err(std::string("KVCacheManager: failed to zero slot: ") +
                                cudaGetErrorString(zero_err));
    }

    seq_to_slot_[seq_id] = slot_idx;
    if (seq_id >= next_seq_id_) {
        next_seq_id_ = seq_id + 1;
    }

    return Result<int>::ok(seq_id);
}

Result<void> KVCacheManager::releaseSequence(int seq_id) {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        TLLM_WARN("releaseSequence: sequence not found: {}", seq_id);
        return Result<void>::err("Sequence not found: " + std::to_string(seq_id));
    }

    int slot_idx = it->second;
    slots_[slot_idx].active = false;
    slots_[slot_idx].seq_id = -1;
    slots_[slot_idx].current_len = 0;

    seq_to_slot_.erase(it);

    TLLM_DEBUG("KVCache: released sequence {} (slot {})", seq_id, slot_idx);
    return Result<void>::ok();
}

Result<std::pair<half *, half *>> KVCacheManager::getCacheChecked(int seq_id, int layer_idx) {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return Result<std::pair<half *, half *>>::err("Sequence not found: " +
                                                      std::to_string(seq_id));
    }

    auto layer_result = Validator::validateLayerIndex(layer_idx, config_.num_layers, "getCache");
    if (layer_result.isErr()) {
        return Result<std::pair<half *, half *>>::err(layer_result.error());
    }

    int    slot_idx = it->second;
    size_t k_offset = calculateOffset(slot_idx, layer_idx, false);
    size_t v_offset = calculateOffset(slot_idx, layer_idx, true);

    return Result<std::pair<half *, half *>>::ok(
        {memory_pool_ + k_offset, memory_pool_ + v_offset});
}

std::pair<half *, half *> KVCacheManager::getCache(int seq_id, int layer_idx) {
    auto result = getCacheChecked(seq_id, layer_idx);
    if (result.isErr()) {
        TLLM_WARN("getCache: {}", result.error());
        return {nullptr, nullptr};
    }
    return result.value();
}

Result<void> KVCacheManager::appendKV(int seq_id, int layer_idx, const half *new_k,
                                      const half *new_v, int num_tokens, cudaStream_t stream) {
    // Validate pointers
    auto ptr_result = Validator::validateNotNull(new_k, "new_k");
    if (ptr_result.isErr()) {
        TLLM_ERROR("appendKV: {}", ptr_result.error());
        return ptr_result;
    }
    ptr_result = Validator::validateNotNull(new_v, "new_v");
    if (ptr_result.isErr()) {
        TLLM_ERROR("appendKV: {}", ptr_result.error());
        return ptr_result;
    }

    // Validate num_tokens
    if (num_tokens <= 0) {
        TLLM_ERROR("appendKV: invalid num_tokens: {}", num_tokens);
        return Result<void>::err("appendKV: num_tokens must be positive: " +
                                 std::to_string(num_tokens));
    }

    // Validate layer index
    auto layer_result = Validator::validateLayerIndex(layer_idx, config_.num_layers, "appendKV");
    if (layer_result.isErr()) {
        TLLM_ERROR("{}", layer_result.error());
        return layer_result;
    }

    // Find sequence
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        TLLM_ERROR("appendKV: sequence not found: {}", seq_id);
        return Result<void>::err("appendKV: sequence not found: " + std::to_string(seq_id));
    }

    int   slot_idx = it->second;
    auto &slot = slots_[slot_idx];

    // Write position is always current_len — all layers see the same value
    // because advanceSeqLen() is called ONCE after all layers have appended.
    int write_pos = slot.current_len;

    // Check if we have space
    if (write_pos + num_tokens > slot.max_len) {
        TLLM_ERROR("appendKV: cache overflow. seq_id={}, write_pos={}, num_tokens={}, max_len={}",
                   seq_id, write_pos, num_tokens, slot.max_len);
        return Result<void>::err("appendKV: cache overflow. Current: " + std::to_string(write_pos) +
                                 ", appending: " + std::to_string(num_tokens) +
                                 ", max: " + std::to_string(slot.max_len));
    }

    // Calculate destination offsets
    auto [k_cache, v_cache] = getCache(seq_id, layer_idx);
    if (!k_cache || !v_cache) {
        TLLM_ERROR("appendKV: failed to get cache pointers");
        return Result<void>::err("appendKV: failed to get cache pointers");
    }

    size_t pos_offset = static_cast<size_t>(write_pos) * config_.num_kv_heads * config_.head_dim;
    size_t copy_size =
        static_cast<size_t>(num_tokens) * config_.num_kv_heads * config_.head_dim * sizeof(half);

    CUDA_CHECK(
        cudaMemcpyAsync(k_cache + pos_offset, new_k, copy_size, cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(
        cudaMemcpyAsync(v_cache + pos_offset, new_v, copy_size, cudaMemcpyDeviceToDevice, stream));

    TLLM_TRACE("appendKV: seq_id={}, layer={}, num_tokens={}, write_pos={}", seq_id, layer_idx,
               num_tokens, write_pos);
    return Result<void>::ok();
}

void KVCacheManager::setAppendPos(int pos, cudaStream_t stream) {
    if (append_pos_.data() == nullptr) return;
    // 修复：用成员变量 append_pos_host_ 作为 H2D 源（graph capture 会固化
    // 该指针并在重放时读取当前值；形参 &pos 是栈地址，重放时不可靠）。
    append_pos_host_ = pos;
    append_pos_.copyFromHost(&append_pos_host_, 1, stream);
}

// 任务 3.2：device 写位置版本的 appendKV。
// 写位置由 device int 提供（调用方保证 *device_write_pos == getSeqLen()），
// 使 append 成为可被 CUDA Graph 重放的确定性 device 工作。
Result<void> KVCacheManager::appendKV(int seq_id, int layer_idx, const half *new_k,
                                      const half *new_v, int num_tokens,
                                      const int *device_write_pos, cudaStream_t stream) {
    // 与 host 版本相同的校验（不复制数据，只做不变量检查）
    auto ptr_result = Validator::validateNotNull(new_k, "new_k");
    if (ptr_result.isErr()) {
        TLLM_ERROR("appendKV: {}", ptr_result.error());
        return ptr_result;
    }
    ptr_result = Validator::validateNotNull(new_v, "new_v");
    if (ptr_result.isErr()) {
        TLLM_ERROR("appendKV: {}", ptr_result.error());
        return ptr_result;
    }
    if (num_tokens <= 0) {
        TLLM_ERROR("appendKV: invalid num_tokens: {}", num_tokens);
        return Result<void>::err("appendKV: num_tokens must be positive: " +
                                 std::to_string(num_tokens));
    }
    auto layer_result = Validator::validateLayerIndex(layer_idx, config_.num_layers, "appendKV");
    if (layer_result.isErr()) {
        TLLM_ERROR("{}", layer_result.error());
        return layer_result;
    }
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        TLLM_ERROR("appendKV: sequence not found: {}", seq_id);
        return Result<void>::err("appendKV: sequence not found: " + std::to_string(seq_id));
    }
    if (device_write_pos == nullptr) {
        return Result<void>::err("appendKV: device_write_pos is null");
    }

    // host 侧溢出检查（device 值与 getSeqLen 一致，host 检查不变量即可）
    auto &slot = slots_[it->second];
    int   write_pos = slot.current_len;
    if (write_pos + num_tokens > slot.max_len) {
        TLLM_ERROR("appendKV: cache overflow. seq_id={}, write_pos={}, num_tokens={}, max_len={}",
                   seq_id, write_pos, num_tokens, slot.max_len);
        return Result<void>::err("appendKV: cache overflow. Current: " + std::to_string(write_pos) +
                                 ", appending: " + std::to_string(num_tokens) +
                                 ", max: " + std::to_string(slot.max_len));
    }

    auto [k_cache, v_cache] = getCache(seq_id, layer_idx);
    if (!k_cache || !v_cache) {
        TLLM_ERROR("appendKV: failed to get cache pointers");
        return Result<void>::err("appendKV: failed to get cache pointers");
    }

    kernels::append_kv_at(new_k, new_v, k_cache, v_cache, device_write_pos, num_tokens,
                          config_.num_kv_heads, config_.head_dim, stream);
    return Result<void>::ok();
}

Result<void> KVCacheManager::advanceSeqLen(int seq_id, int num_tokens) {
    if (num_tokens <= 0) {
        TLLM_ERROR("advanceSeqLen: invalid num_tokens: {}", num_tokens);
        return Result<void>::err("advanceSeqLen: num_tokens must be positive: " +
                                 std::to_string(num_tokens));
    }

    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        TLLM_ERROR("advanceSeqLen: sequence not found: {}", seq_id);
        return Result<void>::err("advanceSeqLen: sequence not found: " + std::to_string(seq_id));
    }

    auto &slot = slots_[it->second];
    int   old_len = slot.current_len;

    // Fail loudly instead of silently clamping: silent truncation would let
    // the engine keep generating while the caller believes the cache advanced,
    // producing irreproducible KV corruption at the sequence tail.
    if (num_tokens > slot.max_len - old_len) {
        TLLM_ERROR("advanceSeqLen: overflow. seq_id={}, current={}, advance={}, max={}", seq_id,
                   old_len, num_tokens, slot.max_len);
        return Result<void>::err("advanceSeqLen: overflow. Current: " + std::to_string(old_len) +
                                 ", advance: " + std::to_string(num_tokens) +
                                 ", max: " + std::to_string(slot.max_len));
    }

    slot.current_len = old_len + num_tokens;

    TLLM_TRACE("advanceSeqLen: seq_id={}, {} -> {}", seq_id, old_len, slot.current_len);
    return Result<void>::ok();
}

Result<void> KVCacheManager::validateAppendSpace(int seq_id, int num_tokens) const noexcept {
    if (num_tokens <= 0) {
        return Result<void>::err("validateAppendSpace: num_tokens must be positive: " +
                                 std::to_string(num_tokens));
    }
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return Result<void>::err("Sequence not found: " + std::to_string(seq_id));
    }
    const auto &slot = slots_[it->second];
    if (slot.current_len + num_tokens > slot.max_len) {
        return Result<void>::err("KV cache overflow: current=" + std::to_string(slot.current_len) +
                                 ", append=" + std::to_string(num_tokens) +
                                 ", max=" + std::to_string(slot.max_len));
    }
    return Result<void>::ok();
}

int KVCacheManager::getSeqLen(int seq_id) const noexcept {
    auto it = seq_to_slot_.find(seq_id);
    if (it == seq_to_slot_.end()) {
        return 0;
    }
    return slots_[it->second].current_len;
}

bool KVCacheManager::hasSequence(int seq_id) const noexcept {
    return seq_to_slot_.find(seq_id) != seq_to_slot_.end();
}

size_t KVCacheManager::getUsedMemory() const noexcept {
    int active_count = 0;
    for (const auto &slot : slots_) {
        if (slot.active) {
            active_count++;
        }
    }
    return active_count * slot_size_;
}

size_t KVCacheManager::getTotalMemory() const noexcept { return pool_size_; }

size_t KVCacheManager::getFreeMemory() const noexcept { return getTotalMemory() - getUsedMemory(); }

int KVCacheManager::getActiveSequenceCount() const noexcept {
    int count = 0;
    for (const auto &slot : slots_) {
        if (slot.active) {
            count++;
        }
    }
    return count;
}

} // namespace tiny_llm
