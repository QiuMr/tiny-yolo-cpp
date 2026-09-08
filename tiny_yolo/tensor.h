// Tiny YOLO - Tensor 张量类
#pragma once
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <climits>
#include <new>
#include <vector>
#include <cassert>

// 简单的张量类，NCHW 布局，float32 为主
struct Tensor {
    std::vector<int> shape;   // 维度形状
    std::vector<int> strides; // 步长（元素数，不是字节数）
    float* data = nullptr;
    int ndim = 0;
    int numel = 0;            // 元素总数
    int dtype = 0;            // 0=float32, 1=int64, 2=int32
    bool own_data = false;    // 是否拥有数据内存（常量张量不拥有，指向模型权重区）

    Tensor() = default;
    ~Tensor() { free_data(); }

    // 禁止拷贝，用 move
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& o) noexcept {
        shape = std::move(o.shape);
        strides = std::move(o.strides);
        data = o.data;
        ndim = o.ndim;
        numel = o.numel;
        dtype = o.dtype;
        own_data = o.own_data;
        o.data = nullptr;
        o.own_data = false;
    }

    void free_data() {
        if (own_data && data) {
            free(data);
            data = nullptr;
        }
    }

    // 根据形状分配内存（形状不变时复用旧内存）
    // 失败时抛 std::bad_alloc（由 run() 统一捕获转成错误码），不再返回悬空指针
    void alloc(const std::vector<int>& s) {
        // 用 size_t 累加再检查：numel 是 int，超大张量（32 位进程尤其容易触发）
        // 不能让连乘悄悄回绕成一个小正数，否则后续会按错误长度越界写
        size_t numel64 = 1;
        for (int d : s) {
            if (d != 0) numel64 *= (size_t)(d > 0 ? d : 1);
        }
        if (numel64 > (size_t)INT_MAX) throw std::bad_alloc();
        int new_numel = (int)numel64;
        // 如果已有内存且大小一致，直接复用（避免频繁malloc/free）
        if (own_data && data && new_numel == numel) {
            shape = s;
            ndim = (int)s.size();
            numel = new_numel;
            strides.resize(ndim);
            int stride = 1;
            for (int i = ndim - 1; i >= 0; i--) {
                strides[i] = stride;
                stride *= shape[i];
            }
            return;
        }
        free_data();
        shape = s;
        ndim = (int)s.size();
        numel = new_numel;
        strides.resize(ndim);
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        data = (float*)malloc((size_t)numel * sizeof(float));
        if (!data) {
            // 32 位进程地址空间紧张，大模型的中层张量完全可能分配失败。
            // 抛异常由上层捕获转成错误码，比留一个 nullptr 让算子去写要安全得多
            numel = 0;
            own_data = false;
            throw std::bad_alloc();
        }
        own_data = true;
    }

    // 引用外部数据（不拥有，不释放）
    void reference(const std::vector<int>& s, float* ptr) {
        free_data();
        shape = s;
        ndim = (int)s.size();
        numel = 1;
        for (int d : s) numel *= d;
        strides.resize(ndim);
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        data = ptr;
        own_data = false;
    }

    // 重置形状但不重新分配（用于 Reshape，要求 numel 不变）
    void reshape(const std::vector<int>& s) {
        int new_numel = 1;
        for (int d : s) new_numel *= (d > 0 ? d : 1);
        // 处理 -1（自动推断）
        if (new_numel != numel) {
            for (size_t i = 0; i < s.size(); i++) {
                if (s[i] == -1) {
                    int known = 1;
                    for (size_t j = 0; j < s.size(); j++) {
                        if (j != i) known *= (s[j] > 0 ? s[j] : 1);
                    }
                    shape = s;
                    shape[i] = numel / known;
                    break;
                }
            }
        } else {
            shape = s;
        }
        ndim = (int)shape.size();
        strides.resize(ndim);
        int stride = 1;
        for (int i = ndim - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
    }

    // 填充常量
    void fill(float val) {
        if (data && own_data) {
            for (int i = 0; i < numel; i++) data[i] = val;
        }
    }

    // 索引访问（4维）
    inline float& at(int n, int c, int h, int w) {
        assert(ndim == 4);
        return data[n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
    }

    inline float at(int n, int c, int h, int w) const {
        assert(ndim == 4);
        return data[n * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]];
    }
};

// 简单的内存池：预分配一大块，避免频繁 malloc
class MemoryPool {
public:
    std::vector<float*> blocks;
    std::vector<int> block_sizes;

    ~MemoryPool() {
        for (float* p : blocks) free(p);
    }

    float* alloc(int numel) {
        // 找一个足够大的空闲块
        for (size_t i = 0; i < blocks.size(); i++) {
            if (block_sizes[i] >= numel && blocks[i] != nullptr) {
                float* p = blocks[i];
                blocks[i] = nullptr; // 标记为已用
                return p;
            }
        }
        // 分配新块
        float* p = (float*)malloc(numel * sizeof(float));
        blocks.push_back(p);
        block_sizes.push_back(numel);
        return p;
    }

    void free(float* p) {
        for (size_t i = 0; i < blocks.size(); i++) {
            if (blocks[i] == nullptr && block_sizes[i] > 0) {
                // 找到对应的已用块（通过地址匹配）
                // 简化：直接标记为空闲
            }
        }
        // 简化版：不真正回收，依赖推理结束后统一释放
    }
};
