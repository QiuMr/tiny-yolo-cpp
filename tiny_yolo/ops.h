// Tiny YOLO - 算子实现
// 全部 inline，减小体积
#pragma once
#include "tensor.h"
#include "model_format.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <emmintrin.h>  // SSE2
#include <immintrin.h>  // AVX2
#include <thread>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <functional>
#include <atomic>
#include <unordered_map>

#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

// 前向声明（定义在下方"激活函数"章节）
static inline float fast_exp(float x);

// ============================================================
// 简单线程池（模型加载时创建，推理时复用，避免线程创建/销毁开销）
// 任务发布：原子代数（release/acquire）；唤醒：每 worker 一个自动重置事件
// （即时唤醒、无定时器分辨率陷阱、空闲零 CPU 占用）
// ============================================================
class SimpleThreadPool {
public:
    static SimpleThreadPool& instance() {
        static SimpleThreadPool pool;
        return pool;
    }

    void init(int num_threads) {
        if (workers.size() > 0) return;
        stop_flag.store(false, std::memory_order_relaxed);
        for (int i = 0; i < num_threads; i++) {
            HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr); // 自动重置
            wake_events.push_back(ev);
            workers.emplace_back([this, ev, i] { worker_loop(ev); });
        }
    }

    void shutdown() {
        stop_flag.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (HANDLE h : wake_events) SetEvent(h);
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
        for (HANDLE h : wake_events) CloseHandle(h);
        wake_events.clear();
    }

    // 并行执行 [0, count) 范围的任务，每个任务调用 func(i)
    void parallel_for(int count, std::function<void(int)> func) {
        int nw = (int)workers.size();
        if (nw == 0 || count <= 1) {
            for (int i = 0; i < count; i++) func(i);
            return;
        }

        current_func = &func;
        next_idx.store(0, std::memory_order_relaxed);
        total_count = count;
        done_count.store(0, std::memory_order_relaxed);
        generation.fetch_add(1, std::memory_order_release); // 发布新任务
        for (HANDLE h : wake_events) SetEvent(h);

        // 主线程也参与工作
        int idx;
        while ((idx = next_idx.fetch_add(1, std::memory_order_relaxed)) < count)
            func(idx);

        // 等待全部完成（workers 已被事件唤醒并工作，短暂自旋即可）
        uint32_t spins = 0;
        while (done_count.load(std::memory_order_acquire) < nw) {
            if (++spins % 256 == 0) SwitchToThread();
            else _mm_pause();
        }
    }

    int num_threads() const { return (int)workers.size() + 1; }

private:
    SimpleThreadPool()
        : stop_flag(false), next_idx(0), total_count(0),
          done_count(0), generation(0), current_func(nullptr) {}

    void worker_loop(HANDLE wake_ev) {
        // 基线取"上一代"：即使本线程启动晚于首次任务发布，也不会漏掉任何一代
        uint64_t my_gen = generation.load(std::memory_order_acquire) - 1;
        while (true) {
            bool has_new = (generation.load(std::memory_order_acquire) != my_gen)
                           && current_func != nullptr;
            if (!has_new) {
                if (stop_flag.load(std::memory_order_acquire)) return;
                // 空闲：阻塞等待唤醒（自动重置事件，零 CPU）
                WaitForSingleObject(wake_ev, INFINITE);
                continue;
            }
            if (stop_flag.load(std::memory_order_acquire)) return;
            my_gen = generation.load(std::memory_order_acquire);

            std::function<void(int)>* fn = current_func;
            int idx;
            while ((idx = next_idx.fetch_add(1, std::memory_order_relaxed)) < total_count)
                (*fn)(idx);

            done_count.fetch_add(1, std::memory_order_acq_rel);
        }
    }

    std::vector<std::thread> workers;
    std::vector<HANDLE> wake_events;
    std::atomic<bool> stop_flag;
    std::atomic<int> next_idx;
    int total_count;                      // 仅主线程写，worker 只读
    std::atomic<int> done_count;
    std::atomic<uint64_t> generation;
    std::function<void(int)>* current_func; // 主线程写、发布后 worker 只读
};

// ============================================================
// 工具函数
// ============================================================

// 简单的 GEMM: C = A * B + bias
// A: [M, K], B: [K, N], C: [M, N], bias: [M] 或 nullptr
static inline void gemm(const float* A, const float* B, const float* bias,
                        float* C, int M, int N, int K) {
    // 转置 B 以提高缓存命中率
    float* Bt = (float*)malloc(K * N * sizeof(float));
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            Bt[j * K + i] = B[i * N + j];

    for (int m = 0; m < M; m++) {
        const float* a_row = A + m * K;
        float* c_row = C + m * N;
        for (int n = 0; n < N; n++) {
            const float* b_col = Bt + n * K;
            float sum = bias ? bias[n] : 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a_row[k] * b_col[k];
            }
            c_row[n] = sum;
        }
    }
    free(Bt);
}

// AVX2 优化的 MatMul: C[M,N] = A[M,K] * B[K,N] + bias[N]
// 对 M 做 4 路展开，对 N 做 8 路 AVX2 向量化（B[k*N+n] 连续）
static inline void matmul_avx(const float* A, const float* B, const float* bias,
                                float* C, int M, int N, int K) {
    int N8 = N & ~7;

    int m = 0;
    for (; m + 3 < M; m += 4) {
        const float* a0 = A + (size_t)m * K;
        const float* a1 = a0 + K;
        const float* a2 = a1 + K;
        const float* a3 = a2 + K;
        float* c0 = C + (size_t)m * N;
        float* c1 = c0 + N;
        float* c2 = c1 + N;
        float* c3 = c2 + N;

        int n = 0;
        for (; n < N8; n += 8) {
            __m256 s0 = bias ? _mm256_loadu_ps(bias + n) : _mm256_setzero_ps();
            __m256 s1 = s0, s2 = s0, s3 = s0;
            for (int k = 0; k < K; k++) {
                __m256 bk = _mm256_loadu_ps(B + (size_t)k * N + n);
                s0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k]), bk, s0);
                s1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k]), bk, s1);
                s2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k]), bk, s2);
                s3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k]), bk, s3);
            }
            _mm256_storeu_ps(c0 + n, s0);
            _mm256_storeu_ps(c1 + n, s1);
            _mm256_storeu_ps(c2 + n, s2);
            _mm256_storeu_ps(c3 + n, s3);
        }
        // 剩余的 n（标量）
        for (; n < N; n++) {
            float s0 = bias ? bias[n] : 0;
            float s1 = s0, s2 = s0, s3 = s0;
            for (int k = 0; k < K; k++) {
                float bk = B[(size_t)k * N + n];
                s0 += a0[k] * bk;
                s1 += a1[k] * bk;
                s2 += a2[k] * bk;
                s3 += a3[k] * bk;
            }
            c0[n] = s0; c1[n] = s1; c2[n] = s2; c3[n] = s3;
        }
    }
    // 剩余的 m（标量）
    for (; m < M; m++) {
        const float* a = A + (size_t)m * K;
        float* c = C + (size_t)m * N;
        int n = 0;
        for (; n < N8; n += 8) {
            __m256 s = bias ? _mm256_loadu_ps(bias + n) : _mm256_setzero_ps();
            for (int k = 0; k < K; k++) {
                s = _mm256_fmadd_ps(_mm256_set1_ps(a[k]), _mm256_loadu_ps(B + (size_t)k * N + n), s);
            }
            _mm256_storeu_ps(c + n, s);
        }
        for (; n < N; n++) {
            float s = bias ? bias[n] : 0;
            for (int k = 0; k < K; k++) s += a[k] * B[(size_t)k * N + n];
            c[n] = s;
        }
    }
}

// AVX2 优化的 GEMM：直接输出 [N, M] 布局
// C[n][m] = bias[n] + sum_k A[m][k] * W[n][k]
// 对 K 循环做 8 路向量化，M 循环做 8 路展开（参考 ncnn 的激进展开策略）
static inline void gemm_nc_avx(const float* A, const float* W, const float* bias,
                                 float* C, int M, int N, int K) {
    int K8 = K & ~7;  // K 向下取整到 8 的倍数
    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;
        float b = bias ? bias[n] : 0.0f;

        // 水平求和辅助函数
        auto hsum256 = [](__m256 v) -> float {
            __m128 lo = _mm256_castps256_ps128(v);
            __m128 hi = _mm256_extractf128_ps(v, 1);
            __m128 sum = _mm_add_ps(lo, hi);
            __m128 shuf = _mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2,3,0,1));
            __m128 sums = _mm_add_ps(sum, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);
            return _mm_cvtss_f32(sums);
        };

        // M 循环 8 路展开：w_n[k] 读一次用于 8 个 m
        int m = 0;
        for (; m + 7 < M; m += 8) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;
            __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps();
            __m256 s2=_mm256_setzero_ps(), s3=_mm256_setzero_ps();
            __m256 s4=_mm256_setzero_ps(), s5=_mm256_setzero_ps();
            __m256 s6=_mm256_setzero_ps(), s7=_mm256_setzero_ps();

            // K 循环 8 路向量化（FMA）
            for (int k = 0; k < K8; k += 8) {
                __m256 w = _mm256_loadu_ps(w_n + k);
                s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + k), w, s0);
                s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + k), w, s1);
                s2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + k), w, s2);
                s3 = _mm256_fmadd_ps(_mm256_loadu_ps(a3 + k), w, s3);
                s4 = _mm256_fmadd_ps(_mm256_loadu_ps(a4 + k), w, s4);
                s5 = _mm256_fmadd_ps(_mm256_loadu_ps(a5 + k), w, s5);
                s6 = _mm256_fmadd_ps(_mm256_loadu_ps(a6 + k), w, s6);
                s7 = _mm256_fmadd_ps(_mm256_loadu_ps(a7 + k), w, s7);
            }

            float sum0=hsum256(s0), sum1=hsum256(s1), sum2=hsum256(s2), sum3=hsum256(s3);
            float sum4=hsum256(s4), sum5=hsum256(s5), sum6=hsum256(s6), sum7=hsum256(s7);

            // 处理剩余的 K（标量）
            for (int k = K8; k < K; k++) {
                float w = w_n[k];
                sum0 += a0[k]*w; sum1 += a1[k]*w; sum2 += a2[k]*w; sum3 += a3[k]*w;
                sum4 += a4[k]*w; sum5 += a5[k]*w; sum6 += a6[k]*w; sum7 += a7[k]*w;
            }

            c_n[m]=sum0+b; c_n[m+1]=sum1+b; c_n[m+2]=sum2+b; c_n[m+3]=sum3+b;
            c_n[m+4]=sum4+b; c_n[m+5]=sum5+b; c_n[m+6]=sum6+b; c_n[m+7]=sum7+b;
        }

        // 4 路展开处理剩余
        for (; m + 3 < M; m += 4) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            __m256 s0=_mm256_setzero_ps(), s1=_mm256_setzero_ps();
            __m256 s2=_mm256_setzero_ps(), s3=_mm256_setzero_ps();
            for (int k = 0; k < K8; k += 8) {
                __m256 w = _mm256_loadu_ps(w_n + k);
                s0 = _mm256_add_ps(s0, _mm256_mul_ps(_mm256_loadu_ps(a0 + k), w));
                s1 = _mm256_add_ps(s1, _mm256_mul_ps(_mm256_loadu_ps(a1 + k), w));
                s2 = _mm256_add_ps(s2, _mm256_mul_ps(_mm256_loadu_ps(a2 + k), w));
                s3 = _mm256_add_ps(s3, _mm256_mul_ps(_mm256_loadu_ps(a3 + k), w));
            }
            float sum0=hsum256(s0), sum1=hsum256(s1), sum2=hsum256(s2), sum3=hsum256(s3);
            for (int k = K8; k < K; k++) {
                float w = w_n[k];
                sum0 += a0[k]*w; sum1 += a1[k]*w; sum2 += a2[k]*w; sum3 += a3[k]*w;
            }
            c_n[m]=sum0+b; c_n[m+1]=sum1+b; c_n[m+2]=sum2+b; c_n[m+3]=sum3+b;
        }

        // 处理剩余的 M（标量版本）
        for (; m < M; m++) {
            const float* a_m = A + (size_t)m * K;
            float s = b;
            for (int k = 0; k < K; k++) s += a_m[k] * w_n[k];
            c_n[m] = s;
        }
    }
}

// SSE2 优化的 GEMM：直接输出 [N, M] 布局
// C[n][m] = bias[n] + sum_k A[m][k] * W[n][k]
// 对 K 循环做 4 路向量化，M 循环做 4 路展开
static inline void gemm_nc_sse(const float* A, const float* W, const float* bias,
                                 float* C, int M, int N, int K) {
    int K4 = K & ~3;  // K 向下取整到 4 的倍数
    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;

        // M 循环 4 路展开
        int m = 0;
        for (; m + 3 < M; m += 4) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
            __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

            // K 循环 4 路向量化（FMA）
            for (int k = 0; k < K4; k += 4) {
                __m128 w = _mm_loadu_ps(w_n + k);
                s0 = _mm_fmadd_ps(_mm_loadu_ps(a0 + k), w, s0);
                s1 = _mm_fmadd_ps(_mm_loadu_ps(a1 + k), w, s1);
                s2 = _mm_fmadd_ps(_mm_loadu_ps(a2 + k), w, s2);
                s3 = _mm_fmadd_ps(_mm_loadu_ps(a3 + k), w, s3);
            }

            // 水平求和
            auto hsum = [](__m128 v) -> float {
                __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
                __m128 sums = _mm_add_ps(v, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                return _mm_cvtss_f32(sums);
            };

            float sum0 = hsum(s0), sum1 = hsum(s1), sum2 = hsum(s2), sum3 = hsum(s3);

            // 处理剩余的 K（标量）
            float b = bias ? bias[n] : 0.0f;
            for (int k = K4; k < K; k++) {
                float w = w_n[k];
                sum0 += a0[k] * w;
                sum1 += a1[k] * w;
                sum2 += a2[k] * w;
                sum3 += a3[k] * w;
            }

            c_n[m] = sum0 + b;
            c_n[m+1] = sum1 + b;
            c_n[m+2] = sum2 + b;
            c_n[m+3] = sum3 + b;
        }

        // 处理剩余的 M（标量版本）
        float b = bias ? bias[n] : 0.0f;
        for (; m < M; m++) {
            const float* a_m = A + (size_t)m * K;
            float s = b;
            for (int k = 0; k < K; k++) s += a_m[k] * w_n[k];
            c_n[m] = s;
        }
    }
}

// Optimized GEMM: directly output [N, M] layout (avoids post-transpose)
// C[n][m] = bias[n] + sum_k A[m][k] * W[n][k]
// A: [M, K], W: [N, K], C: [N, M], bias: [N] or nullptr
static inline void gemm_nc(const float* A, const float* W, const float* bias,
                            float* C, int M, int N, int K) {
    // K >= 32 且 M >= 4 时使用 AVX2 版本
    if (K >= 32 && M >= 4) {
        gemm_nc_avx(A, W, bias, C, M, N, K);
        return;
    }
    // K >= 16 且 M >= 4 时使用 SSE 版本
    if (K >= 16 && M >= 4) {
        gemm_nc_sse(A, W, bias, C, M, N, K);
        return;
    }

    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;
        float b = bias ? bias[n] : 0.0f;

        // M 循环 8 路展开：w_n[k] 读一次用于 8 个 m
        int m = 0;
        for (; m + 7 < M; m += 8) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            const float* a4 = A + (size_t)(m+4) * K;
            const float* a5 = A + (size_t)(m+5) * K;
            const float* a6 = A + (size_t)(m+6) * K;
            const float* a7 = A + (size_t)(m+7) * K;
            float s0=b, s1=b, s2=b, s3=b, s4=b, s5=b, s6=b, s7=b;
            for (int k = 0; k < K; k++) {
                float w = w_n[k];
                s0 += a0[k] * w;
                s1 += a1[k] * w;
                s2 += a2[k] * w;
                s3 += a3[k] * w;
                s4 += a4[k] * w;
                s5 += a5[k] * w;
                s6 += a6[k] * w;
                s7 += a7[k] * w;
            }
            c_n[m]=s0; c_n[m+1]=s1; c_n[m+2]=s2; c_n[m+3]=s3;
            c_n[m+4]=s4; c_n[m+5]=s5; c_n[m+6]=s6; c_n[m+7]=s7;
        }
        // 4 路展开处理剩余
        for (; m + 3 < M; m += 4) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            float s0 = b, s1 = b, s2 = b, s3 = b;
            for (int k = 0; k < K; k++) {
                float w = w_n[k];
                s0 += a0[k] * w;
                s1 += a1[k] * w;
                s2 += a2[k] * w;
                s3 += a3[k] * w;
            }
            c_n[m] = s0; c_n[m+1] = s1; c_n[m+2] = s2; c_n[m+3] = s3;
        }
        // 处理剩余的 m
        for (; m < M; m++) {
            const float* a_m = A + (size_t)m * K;
            float s = b;
            for (int k = 0; k < K; k++) s += a_m[k] * w_n[k];
            c_n[m] = s;
        }
    }
}

// ============================================================
// 并行 GEMM：Out[N, M] = W[N, K] * X[K, M] + bias[N]
// X 行主序 [K][M]，W 行主序 [N][K]，Out 行主序 [N][M]
// 输入直接是卷积特征图的内存布局，无需任何转置
// 对输出通道分块（NB=6）并行，m 方向 AVX2+FMA 8 路向量化
// ============================================================
static inline void gemm_nm_core(const float* X, const float* W, const float* bias,
                                 float* Out, int N, int M, int K, bool silu,
                                 int m0 = 0, int m1 = -1) {
    if (m1 < 0) m1 = M;
    const int NB = 6;
    int num_blocks = (N + NB - 1) / NB;
    for (int blk = 0; blk < num_blocks; blk++) {
        int n0 = blk * NB;
        int n1 = std::min(N, n0 + NB);
        int ni = n1 - n0;
        const float* w0 = W + (size_t)n0 * K;
        float* o0 = Out + (size_t)n0 * M;
        int m = m0;
        for (; m + 7 < m1; m += 8) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            const float* xk = X + m;
            for (int k = 0; k < K; k++, xk += M) {
                __m256 xv = _mm256_loadu_ps(xk);
                acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + k), xv, acc0);
                if (ni > 1) acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + K + k), xv, acc1);
                if (ni > 2) acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + 2 * (size_t)K + k), xv, acc2);
                if (ni > 3) acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + 3 * (size_t)K + k), xv, acc3);
                if (ni > 4) acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + 4 * (size_t)K + k), xv, acc4);
                if (ni > 5) acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(w0 + 5 * (size_t)K + k), xv, acc5);
            }
            __m256 vones = _mm256_set1_ps(1.0f);
            for (int i = 0; i < ni; i++) {
                __m256 v = (i == 0) ? acc0 : (i == 1) ? acc1 : (i == 2) ? acc2 :
                           (i == 3) ? acc3 : (i == 4) ? acc4 : acc5;
                if (bias) v = _mm256_add_ps(v, _mm256_set1_ps(bias[n0 + i]));
                if (silu) {
                    // 向量化 fast_exp（Schraudolph），带 ±88 钳制与标量版语义一致
                    __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), v);
                    t = _mm256_min_ps(t, _mm256_set1_ps(88.0f));
                    t = _mm256_max_ps(t, _mm256_set1_ps(-88.0f));
                    t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                    __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                    v = _mm256_div_ps(v, _mm256_add_ps(vones, ex));
                }
                _mm256_storeu_ps(o0 + (size_t)i * M + m, v);
            }
        }
        // m 尾部（标量）
        for (; m < m1; m++) {
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0;
            for (int k = 0; k < K; k++) {
                float xv = X[(size_t)k * M + m];
                s0 += w0[k] * xv;
                if (ni > 1) s1 += w0[(size_t)K + k] * xv;
                if (ni > 2) s2 += w0[2 * (size_t)K + k] * xv;
                if (ni > 3) s3 += w0[3 * (size_t)K + k] * xv;
                if (ni > 4) s4 += w0[4 * (size_t)K + k] * xv;
                if (ni > 5) s5 += w0[5 * (size_t)K + k] * xv;
            }
            float sv[6] = { s0, s1, s2, s3, s4, s5 };
            for (int i = 0; i < ni; i++) {
                float v = sv[i] + (bias ? bias[n0 + i] : 0.0f);
                if (silu) v = v / (1.0f + fast_exp(-v));
                o0[(size_t)i * M + m] = v;
            }
        }
    }
}

static inline void gemm_nm(const float* X, const float* W, const float* bias,
                            float* Out, int N, int M, int K, bool silu = false) {
    long long macs = (long long)N * M * K;
    int workers = SimpleThreadPool::instance().num_threads();
    if (workers <= 2 || macs <= 500000) {
        gemm_nm_core(X, W, bias, Out, N, M, K, silu);
        return;
    }

    // 分块策略（二维：通道块 × M 块）：
    // - X 很大（超过 L2）时通道块保持 6，减少 X 的重复遍历次数
    //   （总流量 = 通道块数 × X大小，细分通道会成倍放大内存带宽压力）
    // - 任务数不足时优先在 M 方向补充切分（各任务读写区间完全独立）
    size_t xbytes = (size_t)K * M * sizeof(float);
    int CH = 6;
    if (xbytes <= (2u << 20)) {
        while (CH > 1 && (N + CH - 1) / CH < workers * 2) CH--;
    }
    int nb = (N + CH - 1) / CH;
    int mb = 1;
    while (nb * mb < workers * 2 && mb < 8 && (M / (mb * 2)) >= 64) mb *= 2;
    int m_chunk = ((M + mb - 1) / mb + 7) & ~7;
    int m_blocks = (M + m_chunk - 1) / m_chunk;
    int tasks = nb * m_blocks;
    if (tasks <= 1) {
        gemm_nm_core(X, W, bias, Out, N, M, K, silu);
        return;
    }

    SimpleThreadPool::instance().parallel_for(tasks, [&](int t) {
        int nk = t / m_blocks;
        int mk = t % m_blocks;
        int n0 = nk * CH;
        int n1 = std::min(N, n0 + CH);
        int ma = mk * m_chunk;
        int mz = std::min(M, ma + m_chunk);
        gemm_nm_core(X, W + (size_t)n0 * K, bias ? bias + n0 : nullptr,
                     Out + (size_t)n0 * M, n1 - n0, M, K, silu, ma, mz);
    });
}

// 线程局部临时缓冲区（用于 col 等中间计算，避免频繁 malloc/free，多线程安全）
static inline float* get_temp_buf(size_t numel) {
    struct Holder {
        float* p = nullptr;
        size_t cap = 0;
        ~Holder() { if (p) free(p); }
    };
    static thread_local Holder buf;
    if (numel > buf.cap) {
        float* np = (float*)realloc(buf.p, numel * sizeof(float));
        if (!np) return nullptr;
        buf.p = np;
        buf.cap = numel;
    }
    return buf.p;
}

// 3x3 s1 p1 专用卷积 v2（参考 ncnn：同时处理两行输出，中间行共享）
// AVX2 优化：每行一次处理 8 个像素
static inline void conv3x3s1p1(const float* input, const float* weight, const float* bias,
                                 float* output, int N, int C_in, int H, int W, int C_out) {
    int outH = H, outW = W;
    int pH = H + 2, pW = W + 2;

    size_t pad_numel = (size_t)C_in * pH * pW;
    float* padded = get_temp_buf(pad_numel);

    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C_in * H * W;
        float* out_n = output + (size_t)n * C_out * outH * outW;

        // 1. padding
        memset(padded, 0, pad_numel * sizeof(float));
        for (int c = 0; c < C_in; c++) {
            const float* in_c = in_n + (size_t)c * H * W;
            float* pad_c = padded + (size_t)c * pH * pW + pW + 1;
            for (int h = 0; h < H; h++) {
                memcpy(pad_c + h * pW, in_c + h * W, W * sizeof(float));
            }
        }

        // 2. 对每个输出通道（多线程：按输出通道块并行）
        const int P_BLOCK = 4;
        int num_blocks = (C_out + P_BLOCK - 1) / P_BLOCK;
        SimpleThreadPool::instance().parallel_for(num_blocks, [&](int block_idx) {
            int p_begin = block_idx * P_BLOCK;
            int p_end = std::min(C_out, p_begin + P_BLOCK);
            for (int p = p_begin; p < p_end; p++) {
            float* out_p = out_n + (size_t)p * outH * outW;
            float b = bias ? bias[p] : 0.0f;
            for (int i = 0; i < outH * outW; i++) out_p[i] = b;

        // 3. 对每个输入通道累加
            for (int q = 0; q < C_in; q++) {
                const float* img = padded + (size_t)q * pH * pW;
                const float* k = weight + ((size_t)p * C_in + q) * 9;
                const float* k0 = k;
                const float* k1 = k + 3;
                const float* k2 = k + 6;

                // 预加载权重到 AVX2 寄存器
                __m256 vk00 = _mm256_set1_ps(k0[0]);
                __m256 vk01 = _mm256_set1_ps(k0[1]);
                __m256 vk02 = _mm256_set1_ps(k0[2]);
                __m256 vk10 = _mm256_set1_ps(k1[0]);
                __m256 vk11 = _mm256_set1_ps(k1[1]);
                __m256 vk12 = _mm256_set1_ps(k1[2]);
                __m256 vk20 = _mm256_set1_ps(k2[0]);
                __m256 vk21 = _mm256_set1_ps(k2[1]);
                __m256 vk22 = _mm256_set1_ps(k2[2]);

                float* outptr = out_p;
                const float* r0 = img;
                const float* r1 = img + pW;
                const float* r2 = img + pW * 2;
                const float* r3 = img + pW * 3;

                int h = 0;
                // 同时处理两行输出
                for (; h + 1 < outH; h += 2) {
                    float* outptr2 = outptr + outW;

                    int w = 0;
                    // AVX2+FMA：每行一次处理 8 个像素
                    for (; w + 7 < outW; w += 8) {
                        // 第一行输出：r0, r1, r2
                        __m256 sum1 = _mm256_setzero_ps();
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w), vk00, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w + 1), vk01, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w + 2), vk02, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w), vk10, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 1), vk11, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 2), vk12, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w), vk20, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 1), vk21, sum1);
                        sum1 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 2), vk22, sum1);
                        _mm256_storeu_ps(outptr + w, _mm256_add_ps(_mm256_loadu_ps(outptr + w), sum1));

                        // 第二行输出：r1, r2, r3（r1, r2 与第一行共享）
                        __m256 sum2 = _mm256_setzero_ps();
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w), vk00, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 1), vk01, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 2), vk02, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w), vk10, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 1), vk11, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 2), vk12, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + w), vk20, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + w + 1), vk21, sum2);
                        sum2 = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + w + 2), vk22, sum2);
                        _mm256_storeu_ps(outptr2 + w, _mm256_add_ps(_mm256_loadu_ps(outptr2 + w), sum2));
                    }
                    // 剩余像素（标量）
                    for (; w < outW; w++) {
                        float s1 = r0[w]*k0[0] + r0[w+1]*k0[1] + r0[w+2]*k0[2]
                                 + r1[w]*k1[0] + r1[w+1]*k1[1] + r1[w+2]*k1[2]
                                 + r2[w]*k2[0] + r2[w+1]*k2[1] + r2[w+2]*k2[2];
                        outptr[w] += s1;
                        float s2 = r1[w]*k0[0] + r1[w+1]*k0[1] + r1[w+2]*k0[2]
                                 + r2[w]*k1[0] + r2[w+1]*k1[1] + r2[w+2]*k1[2]
                                 + r3[w]*k2[0] + r3[w+1]*k2[1] + r3[w+2]*k2[2];
                        outptr2[w] += s2;
                    }

                    // 跳到下一对行
                    r0 += 2 * pW;
                    r1 += 2 * pW;
                    r2 += 2 * pW;
                    r3 += 2 * pW;
                    outptr += 2 * outW;
                }
                // 处理剩余的单行
                for (; h < outH; h++) {
                    int w = 0;
                    for (; w + 7 < outW; w += 8) {
                        __m256 sum = _mm256_setzero_ps();
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w), vk00, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w + 1), vk01, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + w + 2), vk02, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w), vk10, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 1), vk11, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + w + 2), vk12, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w), vk20, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 1), vk21, sum);
                        sum = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + w + 2), vk22, sum);
                        _mm256_storeu_ps(outptr + w, _mm256_add_ps(_mm256_loadu_ps(outptr + w), sum));
                    }
                    for (; w < outW; w++) {
                        float s = r0[w]*k0[0] + r0[w+1]*k0[1] + r0[w+2]*k0[2]
                                + r1[w]*k1[0] + r1[w+1]*k1[1] + r1[w+2]*k1[2]
                                + r2[w]*k2[0] + r2[w+1]*k2[1] + r2[w+2]*k2[2];
                        outptr[w] += s;
                    }
                    r0 += pW; r1 += pW; r2 += pW;
                    outptr += outW;
                }
            }
            } // end parallel_for block
        });
    }
}

// ============================================================
// Winograd F(2,3) 卷积（3x3 s1 p1，减少 2.25x 乘法）
// 变换后的权重缓存（权重指针 -> U），跨帧复用，避免每帧重算
// ============================================================
struct WinogradWeightCache {
    std::unordered_map<const void*, std::vector<float>> map;
    std::mutex mtx;
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        map.clear();
    }
};
static WinogradWeightCache g_winograd_cache;
static inline void clear_winograd_weight_cache() { g_winograd_cache.clear(); }

// 前向声明（定义在下方）
static inline void winograd_weight_transform(const float* g, float* U);
static inline void winograd_input_transform(const float* d, float* V);
static inline void winograd_output_transform(const float* M, float* Y);

// 批量GEMM版本：把Winograd转换成16个小GEMM，直接用 gemm_nm_core
// M[k][C_out, T] = U[k][C_out, C_in] * V[k][C_in, T]，V 天然是 [K=C_in][M=T] 布局，无需转置
// TEMP profiling
static double g_stage_pad = 0, g_stage_vt = 0, g_stage_gemm = 0, g_stage_ot = 0;
static int g_stage_n = 0;
static inline double now_ms_() {
    return (double)GetTickCount64();
}
static inline void conv3x3s1_winograd(const float* input, const float* weight, const float* bias,
                                        float* output, bool apply_silu,
                                        int N, int C_in, int H, int W, int C_out) {
    int outH = H, outW = W;
    int pH = H + 2, pW = W + 2;

    // tile数量
    int h_tiles = (outH + 1) / 2;
    int w_tiles = (outW + 1) / 2;
    int num_tiles = h_tiles * w_tiles;
    int T = num_tiles;

    // 分配 padded 输入（多分配一行，防止边界 tile 越界读）
    // 注意：get_temp_buf 是单块线程局部缓冲（realloc 会移动指针），
    // 必须一次性申请全部工作区，再用偏移切分出各缓冲区
    size_t pad_numel = (size_t)C_in * pH * pW;

    // 中间缓冲区大小
    size_t U_size = (size_t)16 * C_out * C_in;
    size_t V_size = (size_t)16 * C_in * T;
    size_t M_size = (size_t)16 * C_out * T;
    size_t total = pad_numel + pW + U_size + V_size + M_size;
    float* ws = get_temp_buf(total);
    float* padded = ws;
    float* U = ws + pad_numel + pW;
    float* V = U + U_size;
    float* Mbuf = V + V_size;

    // 1. 权重变换 U[16, C_out, C_in] = G * g * G^T（带缓存）
    {
        std::vector<float>* cached = nullptr;
        {
            std::lock_guard<std::mutex> lock(g_winograd_cache.mtx);
            auto it = g_winograd_cache.map.find(weight);
            if (it != g_winograd_cache.map.end()) {
                cached = &it->second;
            } else {
                auto res = g_winograd_cache.map.emplace(weight, std::vector<float>());
                cached = &res.first->second;
                cached->resize(U_size);
                for (int p = 0; p < C_out; p++) {
                    for (int q = 0; q < C_in; q++) {
                        const float* g = weight + ((size_t)p * C_in + q) * 9;
                        float u[16];
                        winograd_weight_transform(g, u);
                        for (int k = 0; k < 16; k++) {
                            (*cached)[((size_t)k * C_out + p) * C_in + q] = u[k];
                        }
                    }
                }
            }
        }
        memcpy(U, cached->data(), U_size * sizeof(float));
    }

    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C_in * H * W;
        float* out_n = output + (size_t)n * C_out * outH * outW;

        // 2. 对输入做 padding（按输入通道并行，各通道内存区间独立）
        {
            SimpleThreadPool& pool_ = SimpleThreadPool::instance();
            bool pmt = (pool_.num_threads() > 1 && pad_numel >= 32768);
            auto pad_body = [&](int c) {
                const float* in_c = in_n + (size_t)c * H * W;
                float* pad_c = padded + (size_t)c * pH * pW + pW + 1;
                float* pad_row0 = padded + (size_t)c * pH * pW;
                // 该通道的两条边界行/列清零由整体 memset 完成，这里只需拷贝主体
                memset(pad_row0, 0, pH * pW * sizeof(float));
                for (int h = 0; h < H; h++) {
                    memcpy(pad_c + h * pW, in_c + h * W, W * sizeof(float));
                }
            };
            if (pmt && C_in >= 4) {
                int qc = std::max(1, C_in / (pool_.num_threads() * 2));
                int qb = (C_in + qc - 1) / qc;
                pool_.parallel_for(qb, [&](int qi) {
                    int q0 = qi * qc, q1 = std::min(C_in, q0 + qc);
                    for (int c = q0; c < q1; c++) pad_body(c);
                });
            } else {
                for (int c = 0; c < C_in; c++) pad_body(c);
            }
        }
        if ((long long)C_in * T > 8192 && C_in >= 2) {
            const int QC = 4;
            int qblocks = (C_in + QC - 1) / QC;
            SimpleThreadPool::instance().parallel_for(qblocks, [&](int qb) {
                int q_begin = qb * QC;
                int q_end = std::min(C_in, q_begin + QC);
                for (int q = q_begin; q < q_end; q++) {
                    const float* img_q = padded + (size_t)q * pH * pW;
                    float* v_q = V + (size_t)q * T;
                    for (int t = 0; t < T; t++) {
                        int ht = t / w_tiles;
                        int wt = t % w_tiles;
                        int oh = ht * 2, ow = wt * 2;
                        const float* d_src = img_q + oh * pW + ow;
                        float d_block[16];
                        for (int r = 0; r < 4; r++)
                            memcpy(d_block + r * 4, d_src + r * pW, 4 * sizeof(float));
                        float v[16];
                        winograd_input_transform(d_block, v);
                        float* dst = v_q;
                        for (int k = 0; k < 16; k++, dst += (size_t)C_in * T) dst[t] = v[k];
                    }
                }
            });
        } else {
            for (int q = 0; q < C_in; q++) {
                const float* img_q = padded + (size_t)q * pH * pW;
                float* v_q = V + (size_t)q * T;
                for (int t = 0; t < T; t++) {
                    int ht = t / w_tiles;
                    int wt = t % w_tiles;
                    int oh = ht * 2, ow = wt * 2;
                    const float* d_src = img_q + oh * pW + ow;
                    float d_block[16];
                    for (int r = 0; r < 4; r++)
                        memcpy(d_block + r * 4, d_src + r * pW, 4 * sizeof(float));
                    float v[16];
                    winograd_input_transform(d_block, v);
                    float* dst = v_q;
                    for (int k = 0; k < 16; k++, dst += (size_t)C_in * T) dst[t] = v[k];
                }
            }
        }

        // 4. 16 个小 GEMM：M[k] = U[k] × V[k]（按 k 并行，内部串行避免嵌套并行）
        SimpleThreadPool& pool = SimpleThreadPool::instance();
        bool gemm_mt = ((long long)16 * C_out * T * C_in > 500000 && pool.num_threads() > 1);
        if (gemm_mt) {
            pool.parallel_for(16, [&](int k) {
                gemm_nm_core(V + (size_t)k * C_in * T,
                             U + (size_t)k * C_out * C_in,
                             nullptr,
                             Mbuf + (size_t)k * C_out * T,
                             C_out, T, C_in, false);
            });
        } else {
            for (int k = 0; k < 16; k++) {
                gemm_nm_core(V + (size_t)k * C_in * T,
                             U + (size_t)k * C_out * C_in,
                             nullptr,
                             Mbuf + (size_t)k * C_out * T,
                             C_out, T, C_in, false);
            }
        }

        // 5. 输出变换 + bias + SiLU + 写回（按输出通道分块并行）
        const int PC = 8;
        int pblocks = (C_out + PC - 1) / PC;
        bool out_mt = ((long long)C_out * T > 8192 && pool.num_threads() > 1);
        auto out_body = [&](int pb) {
            int p_begin = pb * PC;
            int p_end = std::min(C_out, p_begin + PC);
            for (int t = 0; t < T; t++) {
                int ht = t / w_tiles;
                int wt = t % w_tiles;
                int oh = ht * 2, ow = wt * 2;
                for (int p = p_begin; p < p_end; p++) {
                    float m[16], y[4];
                    for (int k = 0; k < 16; k++) {
                        m[k] = Mbuf[((size_t)k * C_out + p) * T + t];
                    }
                    winograd_output_transform(m, y);
                    float b = bias ? bias[p] : 0.0f;
                    for (int i = 0; i < 4; i++) {
                        float val = y[i] + b;
                        if (apply_silu) val = val / (1.0f + fast_exp(-val));
                        y[i] = val;
                    }
                    float* out_p = out_n + (size_t)p * outH * outW;
                    for (int dh = 0; dh < 2 && oh + dh < outH; dh++) {
                        for (int dw = 0; dw < 2 && ow + dw < outW; dw++) {
                            out_p[(oh + dh) * outW + (ow + dw)] = y[dh * 2 + dw];
                        }
                    }
                }
            }
        };
        if (out_mt && pblocks >= 2) {
            pool.parallel_for(pblocks, out_body);
        } else {
            for (int pb = 0; pb < pblocks; pb++) out_body(pb);
        }
    }
}

// 权重变换：U = G * g * G^T，g 是 3x3，U 是 4x4
static inline void winograd_weight_transform(const float* g, float* U) {
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    float tmp[4][3];
    // tmp = G * g：按列 j 循环，tmp[i][j] = sum_k G[i][k] * g[k][j]
    for (int j = 0; j < 3; j++) {
        float g0 = g[0*3+j], g1 = g[1*3+j], g2 = g[2*3+j];
        tmp[0][j] = g0;
        tmp[1][j] = 0.5f * (g0 + g1 + g2);
        tmp[2][j] = 0.5f * (g0 - g1 + g2);
        tmp[3][j] = g2;
    }
    // U = tmp * G^T：按行 i 循环，U[i][j] = sum_k tmp[i][k] * G[j][k]
    for (int i = 0; i < 4; i++) {
        float t0 = tmp[i][0], t1 = tmp[i][1], t2 = tmp[i][2];
        U[i*4+0] = t0;
        U[i*4+1] = 0.5f * (t0 + t1 + t2);
        U[i*4+2] = 0.5f * (t0 - t1 + t2);
        U[i*4+3] = t2;
    }
}

// 输入变换：V = BT * d * BT^T，d 是 4x4，V 是 4x4
static inline void winograd_input_transform(const float* d, float* V) {
    // BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    float tmp[4][4];
    // 行变换：tmp = BT * d
    for (int c = 0; c < 4; c++) {
        float d0 = d[0*4+c], d1 = d[1*4+c], d2 = d[2*4+c], d3 = d[3*4+c];
        tmp[0][c] = d0 - d2;
        tmp[1][c] = d1 + d2;
        tmp[2][c] = -d1 + d2;
        tmp[3][c] = d1 - d3;  // 修正：BT第4行是[0,1,0,-1]，所以是 d1 - d3
    }
    // 列变换：V = tmp * BT^T
    for (int r = 0; r < 4; r++) {
        float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2], t3 = tmp[r][3];
        V[r*4+0] = t0 - t2;
        V[r*4+1] = t1 + t2;
        V[r*4+2] = -t1 + t2;
        V[r*4+3] = t1 - t3;  // 修正：BT^T的第4列是[0,1,0,-1]^T
    }
}

// 输出变换：Y = AT * M * AT^T，M 是 4x4，Y 是 2x2
static inline void winograd_output_transform(const float* M, float* Y) {
    // AT = [[1,1,1,0],[0,1,-1,-1]]
    float tmp[2][4];
    // 行变换：tmp = AT * M
    for (int c = 0; c < 4; c++) {
        float m0 = M[0*4+c], m1 = M[1*4+c], m2 = M[2*4+c], m3 = M[3*4+c];
        tmp[0][c] = m0 + m1 + m2;
        tmp[1][c] = m1 - m2 - m3;
    }
    // 列变换：Y = tmp * AT^T
    for (int r = 0; r < 2; r++) {
        float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2], t3 = tmp[r][3];
        Y[r*2+0] = t0 + t1 + t2;
        Y[r*2+1] = t1 - t2 - t3;
    }
}

// im2col: 把 NCHW 输入转换成列矩阵
// input: [N, C, H, W], output: [N*outH*outW, C*kH*kW]
static inline void im2col(const float* input, float* output,
                           int N, int C, int H, int W,
                           int kH, int kW, int strideH, int strideW,
                           int padH, int padW, int dilationH, int dilationW,
                           int outH, int outW) {
    int col_size = C * kH * kW;
    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C * H * W;
        for (int oh = 0; oh < outH; oh++) {
            for (int ow = 0; ow < outW; ow++) {
                float* col = output + ((n * outH + oh) * outW + ow) * col_size;
                int col_idx = 0;
                for (int c = 0; c < C; c++) {
                    const float* in_c = in_n + (size_t)c * H * W;
                    for (int kh = 0; kh < kH; kh++) {
                        int ih = oh * strideH - padH + kh * dilationH;
                        for (int kw = 0; kw < kW; kw++) {
                            int iw = ow * strideW - padW + kw * dilationW;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                col[col_idx] = in_c[ih * W + iw];
                            } else {
                                col[col_idx] = 0.0f;
                            }
                            col_idx++;
                        }
                    }
                }
            }
        }
    }
}

// im2col（K 外层布局）：output 为 [C*kH*kW, N*outH*outW]，即 [K, M]
// 与 gemm_nm 的 X 输入布局一致，GEMM 阶段无需转置
static inline void im2col_km(const float* input, float* output,
                              int N, int C, int H, int W,
                              int kH, int kW, int strideH, int strideW,
                              int padH, int padW, int dilationH, int dilationW,
                              int outH, int outW) {
    int M = outH * outW;
    SimpleThreadPool& pool_ = SimpleThreadPool::instance();
    bool imt = (pool_.num_threads() > 1 && (size_t)C * kH * kW * M >= 65536);
    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C * H * W;
        float* col_n = output + (size_t)n * C * kH * kW * M;
        auto body = [&](int c) {
            const float* in_c = in_n + (size_t)c * H * W;
            for (int kh = 0; kh < kH; kh++) {
                for (int kw = 0; kw < kW; kw++) {
                    float* row = col_n + (((size_t)c * kH + kh) * kW + kw) * M;
                    for (int oh = 0; oh < outH; oh++) {
                        int ih = oh * strideH - padH + kh * dilationH;
                        const float* in_row = (ih >= 0 && ih < H) ? in_c + (size_t)ih * W : nullptr;
                        int base = oh * outW;
                        if (!in_row) {
                            memset(row + base, 0, outW * sizeof(float));
                        } else {
                            for (int ow = 0; ow < outW; ow++) {
                                int iw = ow * strideW - padW + kw * dilationW;
                                row[base + ow] = (iw >= 0 && iw < W) ? in_row[iw] : 0.0f;
                            }
                        }
                    }
                }
            }
        };
        if (imt && C >= 4) {
            int cc = std::max(1, C / (pool_.num_threads() * 2));
            int cb = (C + cc - 1) / cc;
            pool_.parallel_for(cb, [&](int ci) {
                int c0 = ci * cc, c1 = std::min(C, c0 + cc);
                for (int c = c0; c < c1; c++) body(c);
            });
        } else {
            for (int c = 0; c < C; c++) body(c);
        }
    }
}

// ============================================================
// Conv
// ============================================================
static inline void op_conv(const Tensor& input, const Tensor& weight, const Tensor* bias,
                           Tensor& output, const ConvAttr& attr, bool apply_silu = false) {
    int N = input.shape[0];
    int C_in = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    int C_out = weight.shape[0];
    int kH = attr.kernel_h;
    int kW = attr.kernel_w;
    int group = attr.group;

    int outH = (H + attr.pad_h * 2 - (attr.dilation_h * (kH - 1) + 1)) / attr.stride_h + 1;
    int outW = (W + attr.pad_w * 2 - (attr.dilation_w * (kW - 1) + 1)) / attr.stride_w + 1;

    output.alloc({N, C_out, outH, outW});

    if (group == 1) {
        int col_size = C_in * kH * kW;
        int M = outH * outW;

        // 3x3 s1 p1：Winograd（SiLU 融合在输出变换阶段）
        bool is_3x3s1p1 = (kH == 3 && kW == 3 && attr.stride_h == 1 && attr.stride_w == 1
                            && attr.pad_h == 1 && attr.pad_w == 1 && attr.dilation_h == 1 && attr.dilation_w == 1);

        if (is_3x3s1p1) {
            conv3x3s1_winograd(input.data, weight.data, bias ? bias->data : nullptr,
                        output.data, apply_silu, N, C_in, H, W, C_out);
            return;
        }

        // 1x1 s1 p0：输入本身就是 [K=C_in, M] 布局，直接 GEMM，无需转置
        bool is_1x1 = (kH == 1 && kW == 1 && attr.stride_h == 1 && attr.stride_w == 1
                        && attr.pad_h == 0 && attr.pad_w == 0 && attr.dilation_h == 1 && attr.dilation_w == 1);

        if (is_1x1) {
            for (int n = 0; n < N; n++) {
                const float* in_n = input.data + (size_t)n * C_in * M;
                float* out_n = output.data + (size_t)n * C_out * M;
                gemm_nm(in_n, weight.data, bias ? bias->data : nullptr,
                        out_n, C_out, M, C_in, apply_silu);
            }
        } else {
            // 普通卷积（如 3x3 s2）：im2col 成 [K, M] 布局 + gemm_nm
            size_t col_numel = (size_t)N * M * col_size;
            float* col = get_temp_buf(col_numel);
            im2col_km(input.data, col, N, C_in, H, W, kH, kW,
                      attr.stride_h, attr.stride_w, attr.pad_h, attr.pad_w,
                      attr.dilation_h, attr.dilation_w, outH, outW);

            for (int n = 0; n < N; n++) {
                float* col_n = col + (size_t)n * M * col_size;
                float* out_n = output.data + (size_t)n * C_out * M;
                gemm_nm(col_n, weight.data, bias ? bias->data : nullptr,
                        out_n, C_out, M, col_size, apply_silu);
            }
        }
    } else {
        // 分组卷积（逐组处理）
        int C_per_group = C_in / group;
        int C_out_per_group = C_out / group;
        int col_size = C_per_group * kH * kW;
        int M = outH * outW;
        size_t col_numel = (size_t)N * M * col_size;
        size_t group_input_numel = (size_t)N * C_per_group * H * W;
        size_t max_numel = col_numel > group_input_numel ? col_numel : group_input_numel;

        float* col = get_temp_buf(max_numel);
        float* group_input = col + col_numel;  // 复用缓冲区的后半部分（如果够大）
        bool need_separate = (col_numel + group_input_numel > max_numel);
        if (need_separate) {
            // 不够大，单独分配
            group_input = (float*)malloc(group_input_numel * sizeof(float));
        }

        for (int g = 0; g < group; g++) {
            // 提取该组的输入
            for (int n = 0; n < N; n++)
                for (int c = 0; c < C_per_group; c++)
                    memcpy(group_input + (n * C_per_group + c) * H * W,
                           input.data + (n * C_in + g * C_per_group + c) * H * W,
                           H * W * sizeof(float));

            im2col(group_input, col, N, C_per_group, H, W, kH, kW,
                   attr.stride_h, attr.stride_w, attr.pad_h, attr.pad_w,
                   attr.dilation_h, attr.dilation_w, outH, outW);

            // 该组的权重
            const float* group_weight = weight.data + (size_t)g * C_out_per_group * col_size;
            const float* group_bias = bias ? bias->data + g * C_out_per_group : nullptr;

            for (int n = 0; n < N; n++) {
                float* col_n = col + (size_t)n * M * col_size;
                float* out_n = output.data + (size_t)n * C_out * M + (size_t)g * C_out_per_group * M;
                // 使用 gemm_nc 直接输出 [C_out_per_group, M] 布局
                gemm_nc(col_n, group_weight, group_bias, out_n, M, C_out_per_group, col_size);
                if (apply_silu) {
                    for (int i = 0; i < C_out_per_group * M; i++) {
                        float x = out_n[i];
                        out_n[i] = x / (1.0f + fast_exp(-x));
                    }
                }
            }
        }
        if (need_separate) free(group_input);
    }
}

// ============================================================
// 激活函数
// ============================================================
// 快速 exp 近似（Schraudolph方法，精度足够推理用，比expf快很多）
static inline float fast_exp(float x) {
    if (x > 88.0f) return 1e30f;
    if (x < -88.0f) return 0.0f;
    // Schraudolph近似：利用IEEE754浮点数的指数位
    union { float f; int32_t i; } u;
    u.i = (int32_t)(12102203.0f * x + 1064866805.0f);
    return u.f;
}

static inline void op_silu(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    for (int i = 0; i < input.numel; i++) {
        float x = input.data[i];
        // SiLU = x * sigmoid(x) = x / (1 + exp(-x))
        output.data[i] = x / (1.0f + fast_exp(-x));
    }
}

static inline void op_sigmoid(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    for (int i = 0; i < input.numel; i++) {
        output.data[i] = 1.0f / (1.0f + fast_exp(-input.data[i]));
    }
}

static inline void op_relu(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    for (int i = 0; i < input.numel; i++) {
        output.data[i] = input.data[i] > 0 ? input.data[i] : 0;
    }
}

static inline void op_exp(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    for (int i = 0; i < input.numel; i++) {
        output.data[i] = fast_exp(input.data[i]);
    }
}

// ============================================================
// 元素级运算（支持广播）
// ============================================================
static inline void broadcast_shape(const std::vector<int>& a, const std::vector<int>& b,
                                    std::vector<int>& out) {
    int max_len = std::max((int)a.size(), (int)b.size());
    std::vector<int> a_pad(max_len - a.size(), 1);
    a_pad.insert(a_pad.end(), a.begin(), a.end());
    std::vector<int> b_pad(max_len - b.size(), 1);
    b_pad.insert(b_pad.end(), b.begin(), b.end());
    out.resize(max_len);
    for (int i = 0; i < max_len; i++) {
        out[i] = std::max(a_pad[i], b_pad[i]);
    }
}

static inline int broadcast_index(int flat_idx, const std::vector<int>& out_shape,
                                   const std::vector<int>& in_shape) {
    // 把输出的扁平索引转换为输入的扁平索引（处理广播）
    int result = 0;
    int stride = 1;
    int offset = (int)out_shape.size() - (int)in_shape.size();
    for (int i = (int)out_shape.size() - 1; i >= 0; i--) {
        int dim = flat_idx % out_shape[i];
        flat_idx /= out_shape[i];
        if (i >= offset) {
            int in_dim = in_shape[i - offset];
            if (in_dim > 1) result += dim * stride;
        }
        stride *= (i >= offset && in_shape[i - offset] > 1) ? in_shape[i - offset] : 1;
    }
    return result;
}

static inline float read_tensor_val(const Tensor& t, int idx) {
    if (t.dtype == 7) { // INT64
        return (float)((const int64_t*)t.data)[idx];
    } else if (t.dtype == 6) { // INT32
        return (float)((const int32_t*)t.data)[idx];
    }
    return t.data[idx];
}

static inline void op_elementwise(const Tensor& a, const Tensor& b, Tensor& output,
                                   char op) {
    std::vector<int> out_shape;
    broadcast_shape(a.shape, b.shape, out_shape);
    output.alloc(out_shape);

    bool a_scalar = a.numel == 1;
    bool b_scalar = b.numel == 1;

    if (a_scalar && b_scalar) {
        float va = a.data[0], vb = b.data[0];
        float r = 0;
        switch (op) { case '+': r = va + vb; break; case '-': r = va - vb; break;
                       case '*': r = va * vb; break; case '/': r = va / vb; break; }
        output.fill(r);
        return;
    }

    // 快速路径 1：形状相同，直接逐元素运算
    if (a.ndim == b.ndim && a.shape == b.shape) {
        const float* pa = a.data;
        const float* pb = b.data;
        float* po = output.data;
        int n = output.numel;
        if (a.dtype == 0 && b.dtype == 0) {
            switch (op) {
                case '+': for (int i = 0; i < n; i++) po[i] = pa[i] + pb[i]; break;
                case '-': for (int i = 0; i < n; i++) po[i] = pa[i] - pb[i]; break;
                case '*': for (int i = 0; i < n; i++) po[i] = pa[i] * pb[i]; break;
                case '/': for (int i = 0; i < n; i++) po[i] = pa[i] / pb[i]; break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                float va = read_tensor_val(a, i), vb = read_tensor_val(b, i);
                switch (op) { case '+': po[i] = va + vb; break; case '-': po[i] = va - vb; break;
                               case '*': po[i] = va * vb; break; case '/': po[i] = va / vb; break; }
            }
        }
        return;
    }

    // 快速路径 2：a 是标量
    if (a_scalar) {
        float va = read_tensor_val(a, 0);
        const float* pb = b.data;
        float* po = output.data;
        int n = output.numel;
        if (b.dtype == 0) {
            switch (op) {
                case '+': for (int i = 0; i < n; i++) po[i] = va + pb[i]; break;
                case '-': for (int i = 0; i < n; i++) po[i] = va - pb[i]; break;
                case '*': for (int i = 0; i < n; i++) po[i] = va * pb[i]; break;
                case '/': for (int i = 0; i < n; i++) po[i] = va / pb[i]; break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                float vb = read_tensor_val(b, i);
                switch (op) { case '+': po[i] = va + vb; break; case '-': po[i] = va - vb; break;
                               case '*': po[i] = va * vb; break; case '/': po[i] = va / vb; break; }
            }
        }
        return;
    }

    // 快速路径 3：b 是标量
    if (b_scalar) {
        float vb = read_tensor_val(b, 0);
        const float* pa = a.data;
        float* po = output.data;
        int n = output.numel;
        if (a.dtype == 0) {
            switch (op) {
                case '+': for (int i = 0; i < n; i++) po[i] = pa[i] + vb; break;
                case '-': for (int i = 0; i < n; i++) po[i] = pa[i] - vb; break;
                case '*': for (int i = 0; i < n; i++) po[i] = pa[i] * vb; break;
                case '/': for (int i = 0; i < n; i++) po[i] = pa[i] / vb; break;
            }
        } else {
            for (int i = 0; i < n; i++) {
                float va = read_tensor_val(a, i);
                switch (op) { case '+': po[i] = va + vb; break; case '-': po[i] = va - vb; break;
                               case '*': po[i] = va * vb; break; case '/': po[i] = va / vb; break; }
            }
        }
        return;
    }

    // 通用路径：广播
    for (int i = 0; i < output.numel; i++) {
        float va = read_tensor_val(a, broadcast_index(i, out_shape, a.shape));
        float vb = read_tensor_val(b, broadcast_index(i, out_shape, b.shape));
        float r = 0;
        switch (op) {
            case '+': r = va + vb; break;
            case '-': r = va - vb; break;
            case '*': r = va * vb; break;
            case '/': r = va / vb; break;
        }
        output.data[i] = r;
    }
}

// ============================================================
// Concat
// ============================================================
static inline void op_concat(const std::vector<Tensor*>& inputs, Tensor& output, int axis) {
    // 规范化 axis（支持负数）
    int ndim = inputs[0]->ndim;
    if (axis < 0) axis += ndim;

    std::vector<int> out_shape = inputs[0]->shape;
    int total_axis = 0;
    for (auto* inp : inputs) total_axis += inp->shape[axis];
    out_shape[axis] = total_axis;
    output.alloc(out_shape);

    // 计算每个输入在输出中的偏移
    int offset = 0;
    SimpleThreadPool& pool_ = SimpleThreadPool::instance();
    bool cmt = (pool_.num_threads() > 1 && (size_t)output.numel * sizeof(float) >= 262144);
    for (auto* inp : inputs) {
        int in_axis_size = inp->shape[axis];
        // 逐元素复制
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= out_shape[i];
        int inner = 1;
        for (int i = axis + 1; i < ndim; i++) inner *= out_shape[i];

        auto copy_body = [&](int o) {
            for (int c = 0; c < in_axis_size; c++) {
                float* dst = output.data + ((size_t)o * total_axis + offset + c) * inner;
                const float* src = inp->data + ((size_t)o * in_axis_size + c) * inner;
                memcpy(dst, src, inner * sizeof(float));
            }
        };
        if (cmt && outer >= 8) {
            int workers2 = pool_.num_threads();
            int chunk = std::max(1, (outer + workers2 * 2 - 1) / (workers2 * 2));
            int blocks = (outer + chunk - 1) / chunk;
            pool_.parallel_for(blocks, [&](int b) {
                int o0 = b * chunk, o1 = std::min(outer, o0 + chunk);
                for (int o = o0; o < o1; o++) copy_body(o);
            });
        } else {
            for (int o = 0; o < outer; o++) copy_body(o);
        }
        offset += in_axis_size;
    }
}

// ============================================================
// Resize (双线性插值，最近邻也支持)
// ============================================================
static inline void op_resize(const Tensor& input, Tensor& output,
                              int outH, int outW, const char* mode) {
    int N = input.shape[0];
    int C = input.shape[1];
    int H = input.shape[2];
    int W = input.shape[3];
    output.alloc({N, C, outH, outW});

    float scaleH = (float)H / outH;
    float scaleW = (float)W / outW;

    // 特化路径：2 倍最近邻上采样（YOLO 上采样的最常见情况）
    // 每个源行水平复制 2 倍后整行写两遍，内存带宽最优
    if (mode[0] == 'n' && outH == H * 2 && outW == W * 2) {
        for (int n = 0; n < N; n++) {
            for (int c = 0; c < C; c++) {
                const float* src = input.data + ((size_t)n * C + c) * H * W;
                float* dst = output.data + ((size_t)n * C + c) * outH * outW;
                for (int ih = 0; ih < H; ih++) {
                    const float* srow = src + (size_t)ih * W;
                    float* drow = dst + (size_t)ih * 2 * outW;
                    for (int iw = 0; iw < W; iw++) {
                        float v = srow[iw];
                        drow[iw * 2] = v;
                        drow[iw * 2 + 1] = v;
                    }
                    memcpy(drow + outW, drow, outW * sizeof(float));
                }
            }
        }
        return;
    }

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    if (mode[0] == 'n') {
                        // 最近邻
                        int ih = std::min((int)(oh * scaleH), H - 1);
                        int iw = std::min((int)(ow * scaleW), W - 1);
                        output.at(n, c, oh, ow) = input.at(n, c, ih, iw);
                    } else {
                        // 双线性 (ONNX half_pixel / PyTorch align_corners=False)
                        float fh = (oh + 0.5f) * scaleH - 0.5f;
                        float fw = (ow + 0.5f) * scaleW - 0.5f;
                        int h0 = (int)floorf(fh);
                        int w0 = (int)floorf(fw);
                        int h1 = h0 + 1;
                        int w1 = w0 + 1;
                        float lh = fh - h0;
                        float lw = fw - w0;
                        h0 = std::max(h0, 0); h1 = std::min(h1, H - 1);
                        w0 = std::max(w0, 0); w1 = std::min(w1, W - 1);
                        float v00 = input.at(n, c, h0, w0);
                        float v01 = input.at(n, c, h0, w1);
                        float v10 = input.at(n, c, h1, w0);
                        float v11 = input.at(n, c, h1, w1);
                        output.at(n, c, oh, ow) =
                            v00 * (1 - lh) * (1 - lw) + v01 * (1 - lh) * lw +
                            v10 * lh * (1 - lw) + v11 * lh * lw;
                    }
                }
            }
        }
    }
}

// ============================================================
// Transpose
// ============================================================
// AVX2 8x8 float 块转置：in[r][c] -> out[c][r]
// in 行主序（行距 lda），out 行主序（行距 ldb）
static inline void transpose8x8_ps(const float* in, size_t lda, float* out, size_t ldb) {
    __m256 r0 = _mm256_loadu_ps(in + 0 * lda);
    __m256 r1 = _mm256_loadu_ps(in + 1 * lda);
    __m256 r2 = _mm256_loadu_ps(in + 2 * lda);
    __m256 r3 = _mm256_loadu_ps(in + 3 * lda);
    __m256 r4 = _mm256_loadu_ps(in + 4 * lda);
    __m256 r5 = _mm256_loadu_ps(in + 5 * lda);
    __m256 r6 = _mm256_loadu_ps(in + 6 * lda);
    __m256 r7 = _mm256_loadu_ps(in + 7 * lda);
    __m256 t0 = _mm256_unpacklo_ps(r0, r1);
    __m256 t1 = _mm256_unpackhi_ps(r0, r1);
    __m256 t2 = _mm256_unpacklo_ps(r2, r3);
    __m256 t3 = _mm256_unpackhi_ps(r2, r3);
    __m256 t4 = _mm256_unpacklo_ps(r4, r5);
    __m256 t5 = _mm256_unpackhi_ps(r4, r5);
    __m256 t6 = _mm256_unpacklo_ps(r6, r7);
    __m256 t7 = _mm256_unpackhi_ps(r6, r7);
    __m256 w0 = _mm256_permute2f128_ps(t0, t4, 0x20);
    __m256 w1 = _mm256_permute2f128_ps(t1, t5, 0x20);
    __m256 w2 = _mm256_permute2f128_ps(t2, t6, 0x20);
    __m256 w3 = _mm256_permute2f128_ps(t3, t7, 0x20);
    __m256 w4 = _mm256_permute2f128_ps(t0, t4, 0x31);
    __m256 w5 = _mm256_permute2f128_ps(t1, t5, 0x31);
    __m256 w6 = _mm256_permute2f128_ps(t2, t6, 0x31);
    __m256 w7 = _mm256_permute2f128_ps(t3, t7, 0x31);
    _mm256_storeu_ps(out + 0 * ldb, _mm256_shuffle_ps(w0, w2, 0x44));
    _mm256_storeu_ps(out + 1 * ldb, _mm256_shuffle_ps(w0, w2, 0xEE));
    _mm256_storeu_ps(out + 2 * ldb, _mm256_shuffle_ps(w1, w3, 0x44));
    _mm256_storeu_ps(out + 3 * ldb, _mm256_shuffle_ps(w1, w3, 0xEE));
    _mm256_storeu_ps(out + 4 * ldb, _mm256_shuffle_ps(w4, w6, 0x44));
    _mm256_storeu_ps(out + 5 * ldb, _mm256_shuffle_ps(w4, w6, 0xEE));
    _mm256_storeu_ps(out + 6 * ldb, _mm256_shuffle_ps(w5, w7, 0x44));
    _mm256_storeu_ps(out + 7 * ldb, _mm256_shuffle_ps(w5, w7, 0xEE));
}

static inline void op_transpose(const Tensor& input, Tensor& output, const std::vector<int>& perm) {
    std::vector<int> out_shape(perm.size());
    for (size_t i = 0; i < perm.size(); i++) out_shape[i] = input.shape[perm[i]];
    output.alloc(out_shape);

    // 快速路径：3D 矩阵转置 perm=[0,2,1]（注意力块常见）
    if (input.ndim == 3 && perm.size() == 3 && perm[0] == 0 && perm[1] == 2 && perm[2] == 1) {
        int R = input.shape[1], Cc = input.shape[2];
        const float* in3 = input.data;
        float* out3 = output.data;
        SimpleThreadPool& pool3 = SimpleThreadPool::instance();
        bool t3 = (pool3.num_threads() > 1 && (size_t)input.numel >= 65536);
        auto body3 = [&](int r) {
            int r0 = r * 8;
            int c = 0;
            for (; c + 8 <= Cc; c += 8)
                transpose8x8_ps(in3 + (size_t)r0 * Cc + c, Cc, out3 + (size_t)c * R + r0, R);
            for (; c < Cc; c++)
                for (int i = r0; i < std::min(R, r0 + 8); i++)
                    out3[(size_t)c * R + i] = in3[(size_t)i * Cc + c];
            // 行尾不足 8 的列已由上面 c 循环覆盖；这里补行块尾部
        };
        int rb = (R + 7) / 8;
        // 列方向也分块以提升并行度
        int cbt = (Cc + 7) / 8;
        if (t3 && rb * cbt >= 2) {
            int tasks = rb * cbt;
            pool3.parallel_for(tasks, [&](int t) {
                int rr = t / cbt, cc2 = t % cbt;
                int r0 = rr * 8, c0 = cc2 * 8;
                int re = std::min(R, r0 + 8), ce = std::min(Cc, c0 + 8);
                if (re - r0 == 8 && ce - c0 == 8) {
                    transpose8x8_ps(in3 + (size_t)r0 * Cc + c0, Cc, out3 + (size_t)c0 * R + r0, R);
                } else {
                    for (int i = r0; i < re; i++)
                        for (int j = c0; j < ce; j++)
                            out3[(size_t)j * R + i] = in3[(size_t)i * Cc + j];
                }
            });
        } else {
            for (int rr = 0; rr < rb; rr++) body3(rr);
        }
        return;
    }

    // 快速路径：4D 张量的常见转置模式
    if (input.ndim == 4 && perm.size() == 4) {
        int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
        const float* in = input.data;
        float* out = output.data;
        SimpleThreadPool& pool_ = SimpleThreadPool::instance();
        bool tmt = (pool_.num_threads() > 1 && (size_t)input.numel >= 65536);

        // perm = [0,2,3,1]: NCHW -> NHWC（按 (n,h) 输出行块并行）
        if (perm[0]==0 && perm[1]==2 && perm[2]==3 && perm[3]==1) {
            int rows = N * H;
            auto body = [&](int r) {
                int n = r / H, h = r % H;
                const float* in_row = in + (size_t)n * C * H * W + (size_t)h * W;
                float* out_row = out + ((size_t)n * H + h) * W * C;
                // 8x8 块转置：8 通道 × 8 个 w
                int c = 0;
                for (; c + 8 <= C; c += 8) {
                    const float* ib = in_row + (size_t)c * H * W;
                    int w = 0;
                    for (; w + 8 <= W; w += 8) {
                        transpose8x8_ps(ib + w, H * W, out_row + (size_t)w * C + c, C);
                    }
                    for (; w < W; w++) {
                        for (int cc = 0; cc < 8; cc++)
                            out_row[(size_t)w * C + c + cc] = in_row[(size_t)(c + cc) * H * W + w];
                    }
                }
                for (; c < C; c++) {
                    for (int w = 0; w < W; w++)
                        out_row[(size_t)w * C + c] = in_row[(size_t)c * H * W + w];
                }
            };
            if (tmt && rows >= 8) pool_.parallel_for(rows, body);
            else { for (int r = 0; r < rows; r++) body(r); }
            return;
        }
        // perm = [0,3,1,2]: NHWC -> NCHW（N=1 时按通道范围分块）
        if (perm[0]==0 && perm[1]==3 && perm[2]==1 && perm[3]==2) {
            int H2 = input.shape[1], W2 = input.shape[2], C2 = input.shape[3];
            size_t plane = (size_t)H2 * W2;
            // 8x8 块转置：8 个 i(=h*W+w) × 8 通道
            auto body = [&](int n) {
                const float* in_n = in + (size_t)n * plane * C2;
                float* out_n = out + (size_t)n * C2 * plane;
                int c = 0;
                for (; c + 8 <= C2; c += 8) {
                    int i = 0;
                    for (; i + 8 <= (int)plane; i += 8) {
                        transpose8x8_ps(in_n + (size_t)i * C2 + c, C2,
                                        out_n + (size_t)c * plane + i, plane);
                    }
                    for (; i < (int)plane; i++) {
                        for (int cc = 0; cc < 8; cc++)
                            out_n[(size_t)(c + cc) * plane + i] = in_n[(size_t)i * C2 + c + cc];
                    }
                }
                for (; c < C2; c++) {
                    for (int i = 0; i < (int)plane; i++)
                        out_n[(size_t)c * plane + i] = in_n[(size_t)i * C2 + c];
                }
            };
            auto copy_channel = [&](int c) {
                const float* in_ch = in + c;
                float* out_ch = out + (size_t)c * plane;
                for (int i = 0; i < (int)plane; i++) out_ch[i] = in_ch[(size_t)i * C2];
            };
            int workers2 = pool_.num_threads();
            if (tmt && N > 1) {
                pool_.parallel_for(N, body);
            } else if (tmt && C2 >= 8) {
                int cc = std::max(8, (C2 / (workers2 * 2)) & ~7);
                if (cc < 8) cc = 8;
                int cb = (C2 + cc - 1) / cc;
                pool_.parallel_for(cb, [&](int ci) {
                    int c0 = ci * cc, c1 = std::min(C2, c0 + cc);
                    for (int c = c0; c < c1; c++) copy_channel(c);
                });
            } else {
                for (int n = 0; n < N; n++) body(n);
            }
            return;
        }
        // perm = [0,1,3,2]: 转置最后两维（按通道行并行）
        if (perm[0]==0 && perm[1]==1 && perm[3]==2 && perm[2]==3) {
            auto body = [&](int nc) {
                const float* in_ptr = in + (size_t)nc * H * W;
                float* out_ptr = out + (size_t)nc * H * W;
                for (int h = 0; h < H; h++) {
                    for (int w = 0; w < W; w++) {
                        out_ptr[(size_t)w * H + h] = in_ptr[(size_t)h * W + w];
                    }
                }
            };
            int nc_total = N * C;
            if (tmt && nc_total >= 8) pool_.parallel_for(nc_total, body);
            else { for (int nc = 0; nc < nc_total; nc++) body(nc); }
            return;
        }
    }

    // 通用路径：用步长计算（比扁平索引+取模快）
    std::vector<int> in_strides(input.ndim);
    int s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = s; s *= input.shape[i]; }

    std::vector<int> out_strides(output.ndim);
    s = 1;
    for (int i = output.ndim - 1; i >= 0; i--) { out_strides[i] = s; s *= output.shape[i]; }

    // 计算 perm 的逆映射：in_dim_to_out_dim
    std::vector<int> inv_perm(input.ndim);
    for (size_t i = 0; i < perm.size(); i++) inv_perm[perm[i]] = (int)i;

    // 用嵌套循环转置（递归展开）
    // 简化：用多维索引迭代器
    std::vector<int> out_idx(output.ndim, 0);
    for (int i = 0; i < output.numel; i++) {
        // 计算输入扁平索引
        int in_flat = 0;
        for (int d = 0; d < input.ndim; d++) {
            in_flat += out_idx[inv_perm[d]] * in_strides[d];
        }
        output.data[i] = input.data[in_flat];

        // 递增输出多维索引
        for (int d = output.ndim - 1; d >= 0; d--) {
            out_idx[d]++;
            if (out_idx[d] < output.shape[d]) break;
            out_idx[d] = 0;
        }
    }
}

// ============================================================
// Reshape
// ============================================================
static inline void op_reshape(const Tensor& input, Tensor& output, const std::vector<int>& new_shape) {
    // 计算总元素数，处理 -1 和 0（ONNX中0表示保留原维度）
    std::vector<int> shape = new_shape;

    // 先处理0维：从输入中复制对应维度
    for (size_t i = 0; i < shape.size() && i < (size_t)input.ndim; i++) {
        if (shape[i] == 0) {
            shape[i] = input.shape[i];
        }
    }

    int known = 1;
    int neg_idx = -1;
    for (size_t i = 0; i < shape.size(); i++) {
        if (shape[i] == -1) neg_idx = (int)i;
        else known *= shape[i];
    }
    if (neg_idx >= 0 && known > 0) shape[neg_idx] = input.numel / known;

    output.reference(shape, input.data); // 引用同一块内存
    output.own_data = false;
}

// MatMul 并行包装：按 M 行分块并行（块大小对齐到 matmul_avx 的 4 行展开）
static inline void matmul_par(const float* A, const float* B, const float* bias,
                               float* C, int M, int N, int K) {
    long long macs = (long long)M * N * K;
    int workers = SimpleThreadPool::instance().num_threads();
    if (workers > 1 && macs > 800000 && M >= 16) {
        int chunk = (M + workers * 2 - 1) / (workers * 2);
        chunk = std::max(4, (chunk + 3) & ~3);
        int blocks = (M + chunk - 1) / chunk;
        SimpleThreadPool::instance().parallel_for(blocks, [&](int b) {
            int m0 = b * chunk;
            int m1 = std::min(M, m0 + chunk);
            matmul_avx(A + (size_t)m0 * K, B, bias, C + (size_t)m0 * N, m1 - m0, N, K);
        });
    } else {
        matmul_avx(A, B, bias, C, M, N, K);
    }
}

// ============================================================
// MatMul
// ============================================================
static inline void op_matmul(const Tensor& a, const Tensor& b, Tensor& output) {
    // 支持 2D 和高维 batch matmul
    int a_ndim = a.ndim, b_ndim = b.ndim;
    if (a_ndim == 2 && b_ndim == 2) {
        int M = a.shape[0], K = a.shape[1], N = b.shape[1];
        output.alloc({M, N});
        matmul_par(a.data, b.data, nullptr, output.data, M, N, K);
    } else {
        // 高维：把前面的维度当作 batch
        int K = a.shape[a_ndim - 1];
        int M = a.shape[a_ndim - 2];
        int N = b.shape[b_ndim - 1];

        // 计算 batch 数（广播）
        int batch_dims = std::max(a_ndim, b_ndim) - 2;
        std::vector<int> out_shape;
        for (int i = 0; i < batch_dims; i++) {
            int da = (i < a_ndim - 2) ? a.shape[i] : 1;
            int db = (i < b_ndim - 2) ? b.shape[i] : 1;
            out_shape.push_back(std::max(da, db));
        }
        out_shape.push_back(M);
        out_shape.push_back(N);
        output.alloc(out_shape);

        int batch = 1;
        for (int d : out_shape) batch *= d;
        batch /= M * N;

        // 批次数多时按 batch 并行
        long long macs = (long long)batch * M * N * K;
        int workers = SimpleThreadPool::instance().num_threads();
        if (workers > 1 && batch >= 4 && macs > 2000000) {
            SimpleThreadPool::instance().parallel_for(batch, [&](int b_idx) {
                int a_total_batch = 1;
                for (int i = 0; i < a_ndim - 2; i++) a_total_batch *= a.shape[i];
                int b_total_batch = 1;
                for (int i = 0; i < b_ndim - 2; i++) b_total_batch *= b.shape[i];
                int a_batch_idx = b_idx % a_total_batch;
                int b_batch_idx = b_idx % b_total_batch;
                matmul_avx(a.data + (size_t)a_batch_idx * M * K,
                           b.data + (size_t)b_batch_idx * K * N, nullptr,
                           output.data + (size_t)b_idx * M * N, M, N, K);
            });
            return;
        }

        for (int b_idx = 0; b_idx < batch; b_idx++) {
            // 找到 a 和 b 对应的 batch 索引（处理广播）
            // a 可能有 batch 维度，b 可能是 2D（所有 batch 共享）
            int a_batch_idx = b_idx;
            int b_batch_idx = b_idx;
            
            // 计算 a 的总 batch 大小
            int a_total_batch = 1;
            for (int i = 0; i < a_ndim - 2; i++) a_total_batch *= a.shape[i];
            // 计算 b 的总 batch 大小
            int b_total_batch = 1;
            for (int i = 0; i < b_ndim - 2; i++) b_total_batch *= b.shape[i];
            
            // 如果 b 没有 batch 维度（2D），所有 batch 共享同一个 b
            if (b_total_batch == 1) b_batch_idx = 0;
            // 如果 a 没有 batch 维度，所有 batch 共享同一个 a
            if (a_total_batch == 1) a_batch_idx = 0;
            
            const float* a_ptr = a.data + a_batch_idx * M * K;
            const float* b_ptr = b.data + b_batch_idx * K * N;
            float* out_ptr = output.data + b_idx * M * N;
            matmul_avx(a_ptr, b_ptr, nullptr, out_ptr, M, N, K);
        }
    }
}

// ============================================================
// Split
// ============================================================
static inline void op_split(const Tensor& input, std::vector<Tensor*>& outputs, int axis,
                             const std::vector<int>& split_sizes_in = std::vector<int>()) {
    int ndim = input.ndim;
    if (axis < 0) axis += ndim;
    int n_out = (int)outputs.size();

    // 计算每个输出在 axis 维度的大小（优先级：传入的split_sizes > 输出张量已有的形状 > 等分）
    std::vector<int> split_sizes(n_out);
    int total = 0;
    bool use_input_sizes = !split_sizes_in.empty() && (int)split_sizes_in.size() == n_out;
    if (use_input_sizes) {
        for (int o = 0; o < n_out; o++) {
            split_sizes[o] = split_sizes_in[o];
            total += split_sizes[o];
        }
        if (total != input.shape[axis]) use_input_sizes = false;
    }
    if (!use_input_sizes) {
        bool use_output_shape = true;
        total = 0;
        for (int o = 0; o < n_out; o++) {
            if (outputs[o]->ndim > axis && outputs[o]->shape[axis] > 0) {
                split_sizes[o] = outputs[o]->shape[axis];
                total += split_sizes[o];
            } else {
                use_output_shape = false;
                break;
            }
        }
        if (!use_output_shape || total != input.shape[axis]) {
            // 等分
            int split_size = input.shape[axis] / n_out;
            for (int o = 0; o < n_out; o++) split_sizes[o] = split_size;
        }
    }

    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    int inner = 1;
    for (int i = axis + 1; i < ndim; i++) inner *= input.shape[i];

    int offset = 0;
    SimpleThreadPool& pool_ = SimpleThreadPool::instance();
    bool cmt = (pool_.num_threads() > 1 && (size_t)input.numel * sizeof(float) >= 262144);
    for (int o = 0; o < n_out; o++) {
        std::vector<int> shape = input.shape;
        shape[axis] = split_sizes[o];
        outputs[o]->alloc(shape);

        auto copy_body = [&](int i) {
            for (int c = 0; c < split_sizes[o]; c++) {
                const float* src = input.data + ((size_t)i * input.shape[axis] + offset + c) * inner;
                float* dst = outputs[o]->data + ((size_t)i * split_sizes[o] + c) * inner;
                memcpy(dst, src, inner * sizeof(float));
            }
        };
        if (cmt && outer >= 8) {
            int workers2 = pool_.num_threads();
            int chunk = std::max(1, (outer + workers2 * 2 - 1) / (workers2 * 2));
            int blocks = (outer + chunk - 1) / chunk;
            pool_.parallel_for(blocks, [&](int b) {
                int i0 = b * chunk, i1 = std::min(outer, i0 + chunk);
                for (int i = i0; i < i1; i++) copy_body(i);
            });
        } else {
            for (int i = 0; i < outer; i++) copy_body(i);
        }
        offset += split_sizes[o];
    }
}

// ============================================================
// Slice
// ============================================================
static inline void op_slice(const Tensor& input, Tensor& output,
                             const std::vector<int>& starts, const std::vector<int>& ends,
                             const std::vector<int>& axes, const std::vector<int>& steps) {
    std::vector<int> out_shape = input.shape;
    // 归一化 starts/ends：处理负索引和 clamp
    std::vector<int> norm_starts = starts, norm_ends = ends;
    for (size_t i = 0; i < axes.size(); i++) {
        int ax = axes[i] < 0 ? axes[i] + input.ndim : axes[i];
        int dim = input.shape[ax];
        int step = i < steps.size() ? steps[i] : 1;
        // 处理负索引
        if (norm_starts[i] < 0) norm_starts[i] += dim;
        if (norm_ends[i] < 0) norm_ends[i] += dim;
        // clamp
        norm_starts[i] = std::max(0, std::min(norm_starts[i], dim));
        norm_ends[i] = std::max(0, std::min(norm_ends[i], dim));
        out_shape[ax] = (norm_ends[i] - norm_starts[i] + step - 1) / step;
    }
    output.alloc(out_shape);

    // 简化实现：逐元素复制
    for (int i = 0; i < output.numel; i++) {
        // 输出扁平索引 -> 输出多维索引
        int idx = i;
        std::vector<int> out_idx(output.ndim);
        for (int d = output.ndim - 1; d >= 0; d--) {
            out_idx[d] = idx % output.shape[d];
            idx /= output.shape[d];
        }
        // 转换为输入索引
        std::vector<int> in_idx = out_idx;
        for (size_t j = 0; j < axes.size(); j++) {
            int ax = axes[j] < 0 ? axes[j] + input.ndim : axes[j];
            int step = j < steps.size() ? steps[j] : 1;
            in_idx[ax] = norm_starts[j] + out_idx[ax] * step;
        }
        // 计算输入扁平索引
        int in_flat = 0;
        int stride = 1;
        for (int d = input.ndim - 1; d >= 0; d--) {
            in_flat += in_idx[d] * stride;
            stride *= input.shape[d];
        }
        output.data[i] = input.data[in_flat];
    }
}

// ============================================================
// ReduceMax / ReduceSum
// ============================================================
static inline void op_reduce(const Tensor& input, Tensor& output,
                              const std::vector<int>& axes, bool keepdims, bool is_max) {
    std::vector<int> out_shape = input.shape;
    std::vector<bool> reduce_dim(input.ndim, false);
    for (int ax : axes) {
        int a = ax < 0 ? ax + input.ndim : ax;
        reduce_dim[a] = true;
    }

    if (keepdims) {
        for (int i = 0; i < input.ndim; i++)
            if (reduce_dim[i]) out_shape[i] = 1;
    } else {
        std::vector<int> new_shape;
        for (int i = 0; i < input.ndim; i++)
            if (!reduce_dim[i]) new_shape.push_back(out_shape[i]);
        out_shape = new_shape;
    }
    output.alloc(out_shape);

    // 优化实现：用步长和循环嵌套
    // 计算输入步长
    std::vector<int> in_strides(input.ndim);
    int s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = s; s *= input.shape[i]; }

    // 计算输出步长
    std::vector<int> out_strides(output.ndim);
    s = 1;
    for (int i = output.ndim - 1; i >= 0; i--) { out_strides[i] = s; s *= output.shape[i]; }

    // 计算 reduce 维度的总大小和步长
    int reduce_size = 1;
    for (int d = 0; d < input.ndim; d++)
        if (reduce_dim[d]) reduce_size *= input.shape[d];

    // 预计算每个 reduce 索引对应的输入偏移
    // 简化：用递归或迭代枚举 reduce 维度组合
    // 对于常见情况（reduce 最后一维或连续维度），可以更高效

    // 通用实现：用多维索引迭代器
    std::vector<int> out_idx(output.ndim, 0);
    std::vector<int> in_idx(input.ndim, 0);

    for (int i = 0; i < output.numel; i++) {
        // 建立输入索引（非 reduce 维度从输出索引映射）
        int out_d = 0;
        for (int d = 0; d < input.ndim; d++) {
            if (!reduce_dim[d]) {
                in_idx[d] = out_idx[out_d++];
            }
        }

        // 计算基础输入偏移（非 reduce 维度）
        int base_offset = 0;
        for (int d = 0; d < input.ndim; d++) {
            if (!reduce_dim[d]) base_offset += in_idx[d] * in_strides[d];
        }

        // 枚举 reduce 维度的所有组合
        float result = is_max ? -1e30f : 0.0f;

        // 用递归或迭代枚举 reduce 维度
        // 简化：用扁平索引 + 解码（但只对 reduce 维度）
        std::vector<int> red_axes;
        for (int d = 0; d < input.ndim; d++)
            if (reduce_dim[d]) red_axes.push_back(d);

        std::vector<int> red_idx(red_axes.size(), 0);
        for (int r = 0; r < reduce_size; r++) {
            // 计算输入偏移
            int offset = base_offset;
            for (size_t j = 0; j < red_axes.size(); j++) {
                offset += red_idx[j] * in_strides[red_axes[j]];
            }

            float v = input.data[offset];
            if (is_max) result = std::max(result, v);
            else result += v;

            // 递增 reduce 索引
            for (int j = (int)red_axes.size() - 1; j >= 0; j--) {
                red_idx[j]++;
                if (red_idx[j] < input.shape[red_axes[j]]) break;
                red_idx[j] = 0;
            }
        }

        output.data[i] = result;

        // 递增输出索引
        for (int d = output.ndim - 1; d >= 0; d--) {
            out_idx[d]++;
            if (out_idx[d] < output.shape[d]) break;
            out_idx[d] = 0;
        }
    }
}

// ============================================================
// Softmax
// ============================================================
static inline float hmax256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_max_ps(lo, hi);
    __m128 shuf = _mm_movehl_ps(lo, lo);
    lo = _mm_max_ps(lo, shuf);
    shuf = _mm_shuffle_ps(lo, lo, _MM_SHUFFLE(1, 0, 1, 0));
    lo = _mm_max_ss(lo, shuf);
    return _mm_cvtss_f32(lo);
}
static inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_shuffle_ps(s, s, _MM_SHUFFLE(2, 3, 0, 1));
    s = _mm_add_ps(s, shuf);
    shuf = _mm_movehl_ps(shuf, s);
    s = _mm_add_ss(s, shuf);
    return _mm_cvtss_f32(s);
}

static inline void op_softmax(const Tensor& input, Tensor& output, int axis) {
    output.alloc(input.shape);
    if (axis < 0) axis += input.ndim;

    int outer = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    int inner = 1;
    for (int i = axis + 1; i < input.ndim; i++) inner *= input.shape[i];
    int axis_size = input.shape[axis];

    // 快速路径：inner==1（reduce最后一维），连续内存访问
    if (inner == 1) {
        const bool vok = false && axis_size >= 16;
        const __m256 vones = _mm256_set1_ps(1.0f);
        for (int o = 0; o < outer; o++) {
            const float* src = input.data + (size_t)o * axis_size;
            float* dst = output.data + (size_t)o * axis_size;
            if (vok) {
                // AVX2：max / 向量化 Schraudolph exp / sum / 归一化
                __m256 vmax = _mm256_set1_ps(-1e30f);
                int a = 0;
                for (; a + 8 <= axis_size; a += 8)
                    vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + a));
                float max_val = hmax256_ps(vmax);
                for (; a < axis_size; a++) max_val = std::max(max_val, src[a]);
                __m256 vsum = _mm256_setzero_ps();
                const __m256 vlo88 = _mm256_set1_ps(-88.0f), vhi88 = _mm256_set1_ps(88.0f);
                const __m256 vscale = _mm256_set1_ps(12102203.0f), vbias = _mm256_set1_ps(1064866805.0f);
                for (a = 0; a + 8 <= axis_size; a += 8) {
                    __m256 t = _mm256_sub_ps(_mm256_loadu_ps(src + a), _mm256_set1_ps(max_val));
                    t = _mm256_min_ps(_mm256_max_ps(t, vlo88), vhi88);
                    t = _mm256_fmadd_ps(t, vscale, vbias);
                    __m256 e = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                    _mm256_storeu_ps(dst + a, e);
                    vsum = _mm256_add_ps(vsum, e);
                }
                float sum = hsum256_ps(vsum);
                for (; a < axis_size; a++) { dst[a] = fast_exp(src[a] - max_val); sum += dst[a]; }
                __m256 vinv = _mm256_div_ps(vones, _mm256_set1_ps(sum));
                for (a = 0; a + 8 <= axis_size; a += 8)
                    _mm256_storeu_ps(dst + a, _mm256_mul_ps(_mm256_loadu_ps(dst + a), vinv));
                for (; a < axis_size; a++) dst[a] *= 1.0f / sum;
            } else {
                // 找最大值
                float max_val = src[0];
                for (int a = 1; a < axis_size; a++) {
                    if (src[a] > max_val) max_val = src[a];
                }
                // 计算 exp 和 sum
                float sum = 0;
                for (int a = 0; a < axis_size; a++) {
                    float v = fast_exp(src[a] - max_val);
                    dst[a] = v;
                    sum += v;
                }
                // 归一化
                float inv_sum = 1.0f / sum;
                for (int a = 0; a < axis_size; a++) {
                    dst[a] *= inv_sum;
                }
            }
        }
        return;
    }

    for (int o = 0; o < outer; o++) {
        for (int i = 0; i < inner; i++) {
            // 找最大值
            float max_val = -1e30f;
            for (int a = 0; a < axis_size; a++) {
                int idx = ((o * axis_size + a) * inner) + i;
                max_val = std::max(max_val, input.data[idx]);
            }
            // 计算 exp 和 sum
            float sum = 0;
            for (int a = 0; a < axis_size; a++) {
                int idx = ((o * axis_size + a) * inner) + i;
                float v = fast_exp(input.data[idx] - max_val);
                output.data[idx] = v;
                sum += v;
            }
            // 归一化
            float inv_sum = 1.0f / sum;
            for (int a = 0; a < axis_size; a++) {
                int idx = ((o * axis_size + a) * inner) + i;
                output.data[idx] *= inv_sum;
            }
        }
    }
}

// ============================================================
// MaxPool
// ============================================================
static inline void op_maxpool(const Tensor& input, Tensor& output,
                               int kH, int kW, int strideH, int strideW,
                               int padH, int padW) {
    int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
    int outH = (H + padH * 2 - kH) / strideH + 1;
    int outW = (W + padW * 2 - kW) / strideW + 1;
    output.alloc({N, C, outH, outW});

    for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float max_val = -1e30f;
                    for (int kh = 0; kh < kH; kh++) {
                        for (int kw = 0; kw < kW; kw++) {
                            int ih = oh * strideH - padH + kh;
                            int iw = ow * strideW - padW + kw;
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                max_val = std::max(max_val, input.at(n, c, ih, iw));
                            }
                        }
                    }
                    output.at(n, c, oh, ow) = max_val;
                }
            }
        }
    }
}

// ============================================================
// Unsqueeze
// ============================================================
static inline void op_unsqueeze(const Tensor& input, Tensor& output, const std::vector<int>& axes) {
    // 计算输出形状
    std::vector<int> out_shape;
    int in_idx = 0;
    int total_dims = input.ndim + (int)axes.size();
    std::vector<bool> is_unsqueeze(total_dims, false);
    for (int ax : axes) {
        int norm_ax = ax < 0 ? ax + total_dims : ax;
        if (norm_ax >= 0 && norm_ax < total_dims) is_unsqueeze[norm_ax] = true;
    }
    for (int i = 0; i < total_dims; i++) {
        if (is_unsqueeze[i]) out_shape.push_back(1);
        else if (in_idx < input.ndim) out_shape.push_back(input.shape[in_idx++]);
        else out_shape.push_back(1);
    }
    output.alloc(out_shape);
    memcpy(output.data, input.data, input.numel * sizeof(float));
}

// ============================================================
// Flatten
// ============================================================
static inline void op_flatten(const Tensor& input, Tensor& output, int axis) {
    if (axis < 0) axis += input.ndim;
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    for (int i = axis; i < input.ndim; i++) inner *= input.shape[i];
    output.alloc({outer, inner});
    memcpy(output.data, input.data, input.numel * sizeof(float));
}

// ============================================================
// Tile
// ============================================================
static inline void op_tile(const Tensor& input, Tensor& output, const std::vector<int>& repeats) {
    std::vector<int> out_shape(input.ndim);
    for (int i = 0; i < input.ndim; i++) {
        out_shape[i] = input.shape[i] * (i < (int)repeats.size() ? repeats[i] : 1);
    }
    output.alloc(out_shape);

    // 计算输入和输出步长
    std::vector<int> in_strides(input.ndim), out_strides(input.ndim);
    int in_s = 1, out_s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) {
        in_strides[i] = in_s; in_s *= input.shape[i];
        out_strides[i] = out_s; out_s *= out_shape[i];
    }

    // 逐元素复制
    for (int i = 0; i < output.numel; i++) {
        int in_offset = 0;
        int rem = i;
        for (int d = 0; d < input.ndim; d++) {
            int out_idx = rem / out_strides[d];
            rem %= out_strides[d];
            int in_idx = out_idx % input.shape[d];
            in_offset += in_idx * in_strides[d];
        }
        output.data[i] = input.data[in_offset];
    }
}

// ============================================================
// Gather
// ============================================================
static inline void op_gather(const Tensor& input, Tensor& output, const Tensor& indices, int axis) {
    if (axis < 0) axis += input.ndim;
    int num_indices = indices.numel;

    // 计算输出形状
    std::vector<int> out_shape;
    for (int i = 0; i < axis; i++) out_shape.push_back(input.shape[i]);
    for (int i = 0; i < indices.ndim; i++) out_shape.push_back(indices.shape[i]);
    for (int i = axis + 1; i < input.ndim; i++) out_shape.push_back(input.shape[i]);
    output.alloc(out_shape);

    // 计算步长
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    for (int i = axis + 1; i < input.ndim; i++) inner *= input.shape[i];
    int axis_size = input.shape[axis];

    for (int o = 0; o < outer; o++) {
        for (int idx = 0; idx < num_indices; idx++) {
            int in_idx;
            if (indices.dtype == 7) { // INT64
                in_idx = (int)((const int64_t*)indices.data)[idx];
            } else {
                in_idx = (int)indices.data[idx];
            }
            if (in_idx < 0) in_idx += axis_size;
            if (in_idx < 0) in_idx = 0;
            if (in_idx >= axis_size) in_idx = axis_size - 1;
            const float* src = input.data + (o * axis_size + in_idx) * inner;
            float* dst = output.data + (o * num_indices + idx) * inner;
            memcpy(dst, src, inner * sizeof(float));
        }
    }
}

// ============================================================
// GatherElements
// ============================================================
static inline void op_gather_elements(const Tensor& input, Tensor& output, const Tensor& indices, int axis) {
    if (axis < 0) axis += input.ndim;
    output.alloc(indices.shape);

    // 计算步长
    std::vector<int> in_strides(input.ndim), out_strides(indices.ndim);
    int in_s = 1, out_s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = in_s; in_s *= input.shape[i]; }
    for (int i = indices.ndim - 1; i >= 0; i--) { out_strides[i] = out_s; out_s *= indices.shape[i]; }

    for (int i = 0; i < output.numel; i++) {
        // 解码输出索引
        std::vector<int> out_idx(indices.ndim);
        int rem = i;
        for (int d = 0; d < indices.ndim; d++) {
            out_idx[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        // 计算输入偏移（axis维度用indices的值）
        int in_offset = 0;
        for (int d = 0; d < input.ndim; d++) {
            int idx;
            if (d == axis) {
                if (indices.dtype == 7) { // INT64
                    idx = (int)((const int64_t*)indices.data)[i];
                } else {
                    idx = (int)indices.data[i];
                }
                if (idx < 0) idx += input.shape[d];
            } else {
                idx = out_idx[d];
            }
            if (idx < 0) idx = 0;
            if (idx >= input.shape[d]) idx = input.shape[d] - 1;
            in_offset += idx * in_strides[d];
        }
        output.data[i] = input.data[in_offset];
    }
}

// ============================================================
// TopK
// ============================================================
static inline void op_topk(const Tensor& input, Tensor& output_values, Tensor& output_indices,
                            int k, int axis, bool largest, bool sorted) {
    if (axis < 0) axis += input.ndim;
    int axis_size = input.shape[axis];
    if (k > axis_size) k = axis_size;

    // 输出形状
    std::vector<int> out_shape = input.shape;
    out_shape[axis] = k;
    output_values.alloc(out_shape);
    output_indices.alloc(out_shape);

    // 计算步长
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    for (int i = axis + 1; i < input.ndim; i++) inner *= input.shape[i];

    for (int o = 0; o < outer; o++) {
        for (int inn = 0; inn < inner; inn++) {
            // 收集axis维度的所有值
            std::vector<std::pair<float, int>> vec(axis_size);
            for (int a = 0; a < axis_size; a++) {
                int offset = (o * axis_size + a) * inner + inn;
                vec[a] = {input.data[offset], a};
            }
            // 排序
            if (largest) {
                std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            // 取前k个
            for (int i = 0; i < k; i++) {
                int out_offset = (o * k + i) * inner + inn;
                output_values.data[out_offset] = vec[i].first;
                output_indices.data[out_offset] = (float)vec[i].second;
            }
        }
    }
}

// ============================================================
// Mod
// ============================================================
static inline void op_mod(const Tensor& a, const Tensor& b, Tensor& output) {
    std::vector<int> out_shape;
    broadcast_shape(a.shape, b.shape, out_shape);
    output.alloc(out_shape);

    bool a_scalar = a.numel == 1;
    bool b_scalar = b.numel == 1;

    if (a.ndim == b.ndim && a.shape == b.shape) {
        for (int i = 0; i < output.numel; i++) {
            float va = read_tensor_val(a, i), vb = read_tensor_val(b, i);
            output.data[i] = va - vb * floorf(va / vb);
        }
        return;
    }

    // 通用广播
    for (int i = 0; i < output.numel; i++) {
        float va = read_tensor_val(a, broadcast_index(i, out_shape, a.shape));
        float vb = read_tensor_val(b, broadcast_index(i, out_shape, b.shape));
        output.data[i] = va - vb * floorf(va / vb);
    }
}

// ============================================================
// Cast (目前只支持float，直接复制)
// ============================================================
static inline void op_cast(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    memcpy(output.data, input.data, input.numel * sizeof(float));
}

