// Tiny YOLO - ����ʵ��
// ȫ�� inline����С���?
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

// ǰ���������������·�"�����"�½ڣ�
static inline float fast_exp(float x);

// ============================================================
// ���̳߳أ�ģ�ͼ���ʱ����������ʱ���ã������̴߳���/���ٿ�����
// ���񷢲���ԭ�Ӵ�����release/acquire�������ѣ�ÿ worker һ���Զ������¼�
// ����ʱ���ѡ��޶�ʱ���ֱ������塢������ CPU ռ�ã�
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
            HANDLE ev = CreateEventW(nullptr, FALSE, FALSE, nullptr); // �Զ�����
            wake_events.push_back(ev);
            workers.emplace_back([this, ev, i] { worker_loop(ev); });
        }
    }

    void shutdown() {
        if (workers.empty()) return;
        stop_flag.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (HANDLE h : wake_events) SetEvent(h);
        for (auto& w : workers) {
            if (w.joinable()) w.join();
        }
        workers.clear();
        for (HANDLE h : wake_events) CloseHandle(h);
        wake_events.clear();
        stop_flag.store(false, std::memory_order_relaxed);
        generation.store(0, std::memory_order_relaxed);
        current_func = nullptr;
    }

    // �����˳�ʱ���ã�ֻ���źŲ� detach�������� DllMain
    void shutdown_detach() {
        if (workers.empty()) return;
        stop_flag.store(true, std::memory_order_release);
        generation.fetch_add(1, std::memory_order_release);
        for (HANDLE h : wake_events) SetEvent(h);
        for (auto& w : workers) {
            if (w.joinable()) w.detach();
        }
        workers.clear();
        for (HANDLE h : wake_events) CloseHandle(h);
        wake_events.clear();
    }

    ~SimpleThreadPool() {
        if (!workers.empty()) {
            // ��̬��������ʱ���� join�������� loader lock �ڣ���ֱ�� detach ���� std::terminate
            stop_flag.store(true, std::memory_order_release);
            generation.fetch_add(1, std::memory_order_release);
            for (HANDLE h : wake_events) SetEvent(h);
            for (auto& w : workers) if (w.joinable()) w.detach();
            for (HANDLE h : wake_events) CloseHandle(h);
        }
    }

    // ����ִ�� [0, count) ��Χ������ÿ���������?func(i)
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
        generation.fetch_add(1, std::memory_order_release); // ����������
        for (HANDLE h : wake_events) SetEvent(h);

        // ���߳�Ҳ���빤��
        int idx;
        while ((idx = next_idx.fetch_add(1, std::memory_order_relaxed)) < count)
            func(idx);

        // �ȴ�ȫ����ɣ�workers �ѱ��¼����Ѳ������������������ɣ�
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
        // ����ȡ"��һ��"����ʹ���߳����������״����񷢲���Ҳ����©���κ�һ��
        uint64_t my_gen = generation.load(std::memory_order_acquire) - 1;
        while (true) {
            bool has_new = (generation.load(std::memory_order_acquire) != my_gen)
                           && current_func != nullptr;
            if (!has_new) {
                if (stop_flag.load(std::memory_order_acquire)) return;
                // ���У������ȴ����ѣ��Զ������¼����� CPU��
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
    int total_count;                      // �����߳�д��worker ֻ��
    std::atomic<int> done_count;
    std::atomic<uint64_t> generation;
    std::function<void(int)>* current_func; // ���߳�д�������� worker ֻ��
};

// ============================================================
// ���ߺ���
// ============================================================

// �򵥵� GEMM: C = A * B + bias
// A: [M, K], B: [K, N], C: [M, N], bias: [M] �� nullptr
static inline void gemm(const float* A, const float* B, const float* bias,
                        float* C, int M, int N, int K) {
    // ת�� B ����߻���������?
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

// AVX2 �Ż��� MatMul: C[M,N] = A[M,K] * B[K,N] + bias[N]
// �� M �� 4 ·չ������ N �� 8 · AVX2 ��������B[k*N+n] ������
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
        // ʣ���?n��������
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
    // ʣ���?m��������
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

// AVX2 �Ż��� GEMM��ֱ�����?[N, M] ����
// C[n][m] = bias[n] + sum_k A[m][k] * W[n][k]
// �� K ѭ���� 8 ·��������M ѭ���� 8 ·չ�����ο� ncnn �ļ���չ�����ԣ�
static inline void gemm_nc_avx(const float* A, const float* W, const float* bias,
                                 float* C, int M, int N, int K) {
    int K8 = K & ~7;  // K ����ȡ���� 8 �ı���
    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;
        float b = bias ? bias[n] : 0.0f;

        // ˮƽ��͸�������?
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

        // M ѭ�� 8 ·չ����w_n[k] ��һ������ 8 �� m
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

            // K ѭ�� 8 ·��������FMA��
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

            // ����ʣ���?K��������
            for (int k = K8; k < K; k++) {
                float w = w_n[k];
                sum0 += a0[k]*w; sum1 += a1[k]*w; sum2 += a2[k]*w; sum3 += a3[k]*w;
                sum4 += a4[k]*w; sum5 += a5[k]*w; sum6 += a6[k]*w; sum7 += a7[k]*w;
            }

            c_n[m]=sum0+b; c_n[m+1]=sum1+b; c_n[m+2]=sum2+b; c_n[m+3]=sum3+b;
            c_n[m+4]=sum4+b; c_n[m+5]=sum5+b; c_n[m+6]=sum6+b; c_n[m+7]=sum7+b;
        }

        // 4 ·չ������ʣ��
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

        // ����ʣ���?M�������汾��
        for (; m < M; m++) {
            const float* a_m = A + (size_t)m * K;
            float s = b;
            for (int k = 0; k < K; k++) s += a_m[k] * w_n[k];
            c_n[m] = s;
        }
    }
}

// SSE2 �Ż��� GEMM��ֱ�����?[N, M] ����
// C[n][m] = bias[n] + sum_k A[m][k] * W[n][k]
// �� K ѭ���� 4 ·��������M ѭ���� 4 ·չ��
static inline void gemm_nc_sse(const float* A, const float* W, const float* bias,
                                 float* C, int M, int N, int K) {
    int K4 = K & ~3;  // K ����ȡ���� 4 �ı���
    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;

        // M ѭ�� 4 ·չ��
        int m = 0;
        for (; m + 3 < M; m += 4) {
            const float* a0 = A + (size_t)m * K;
            const float* a1 = A + (size_t)(m+1) * K;
            const float* a2 = A + (size_t)(m+2) * K;
            const float* a3 = A + (size_t)(m+3) * K;
            __m128 s0 = _mm_setzero_ps(), s1 = _mm_setzero_ps();
            __m128 s2 = _mm_setzero_ps(), s3 = _mm_setzero_ps();

            // K ѭ�� 4 ·��������FMA��
            for (int k = 0; k < K4; k += 4) {
                __m128 w = _mm_loadu_ps(w_n + k);
                s0 = _mm_fmadd_ps(_mm_loadu_ps(a0 + k), w, s0);
                s1 = _mm_fmadd_ps(_mm_loadu_ps(a1 + k), w, s1);
                s2 = _mm_fmadd_ps(_mm_loadu_ps(a2 + k), w, s2);
                s3 = _mm_fmadd_ps(_mm_loadu_ps(a3 + k), w, s3);
            }

            // ˮƽ���?
            auto hsum = [](__m128 v) -> float {
                __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2,3,0,1));
                __m128 sums = _mm_add_ps(v, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                return _mm_cvtss_f32(sums);
            };

            float sum0 = hsum(s0), sum1 = hsum(s1), sum2 = hsum(s2), sum3 = hsum(s3);

            // ����ʣ���?K��������
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

        // ����ʣ���?M�������汾��
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
    // K >= 32 �� M >= 4 ʱʹ�� AVX2 �汾
    if (K >= 32 && M >= 4) {
        gemm_nc_avx(A, W, bias, C, M, N, K);
        return;
    }
    // K >= 16 �� M >= 4 ʱʹ�� SSE �汾
    if (K >= 16 && M >= 4) {
        gemm_nc_sse(A, W, bias, C, M, N, K);
        return;
    }

    for (int n = 0; n < N; n++) {
        const float* w_n = W + (size_t)n * K;
        float* c_n = C + (size_t)n * M;
        float b = bias ? bias[n] : 0.0f;

        // M ѭ�� 8 ·չ����w_n[k] ��һ������ 8 �� m
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
        // 4 ·չ������ʣ��
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
        // ����ʣ���?m
        for (; m < M; m++) {
            const float* a_m = A + (size_t)m * K;
            float s = b;
            for (int k = 0; k < K; k++) s += a_m[k] * w_n[k];
            c_n[m] = s;
        }
    }
}

// ============================================================
// ���� GEMM��Out[N, M] = W[N, K] * X[K, M] + bias[N]
// X ������ [K][M]��W ������ [N][K]��Out ������ [N][M]
// ����ֱ���Ǿ�������ͼ���ڴ沼�֣������κ�ת��
// �����ͨ���ֿ飨NB=6�����У�m ���� AVX2+FMA 8 ·������
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
                    // 向量化 fast_exp（Schraudolph）：钳制必须远离 -87.99 毒区
                    // （x < -87.9895 时整数结果变负，reinterpret 成 float 即 NaN）
                    __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), v);
                    t = _mm256_min_ps(t, _mm256_set1_ps(86.0f));
                    t = _mm256_max_ps(t, _mm256_set1_ps(-86.0f));
                    t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                    __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                    v = _mm256_div_ps(v, _mm256_add_ps(vones, ex));
                }
                _mm256_storeu_ps(o0 + (size_t)i * M + m, v);
            }
        }
        // m β����������
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

// W ������棺��?W[N,K] Ԥ��Ϊ [K][N/6][6] �����飬FMA ʱ 6 ��ͨ��ֵ��������ͬһ������
struct PackedW { std::vector<float> data; int N = 0, K = 0; };
static std::unordered_map<const void*, PackedW> g_packedW;
static std::mutex g_packedW_mtx;

static inline const float* get_packed_W(const float* W, int N, int K) {
    if (N % 6 != 0 || K < 32) return nullptr; // ���Դ� K �� N Ϊ 6 ������ 1x1 �������?
    std::lock_guard<std::mutex> lk(g_packedW_mtx);
    auto it = g_packedW.find(W);
    if (it != g_packedW.end() && it->second.N == N && it->second.K == K) return it->second.data.data();
    // ��Ȩ�ػ���״�仯����ģ�����غ��ַ���ã��������ؽ������Ǿ����?
    PackedW pw; pw.N = N; pw.K = K;
    pw.data.resize((size_t)K * N);
    // ����: [K][nb][6]��k*Nb*6 + nb*6 + i
    int nb = N / 6;
    for (int k = 0; k < K; k++) {
        for (int b = 0; b < nb; b++) {
            for (int i = 0; i < 6; i++) {
                pw.data[(size_t)k * N + b * 6 + i] = W[(b * 6 + i) * K + k];
            }
        }
    }
    g_packedW[W] = std::move(pw);
    return g_packedW[W].data.data();
}

static inline void clear_packed_W_cache() {
    std::lock_guard<std::mutex> lk(g_packedW_mtx);
    g_packedW.clear();
}

static inline void gemm_nm_packed_core(const float* X, const float* Wp, const float* bias,
                                        float* Out, int N, int M, int K, bool silu,
                                        int m0, int m1) {
    // Wp �Ѵ���?[K][N] ת��������Wp[k*N + n]
    int nb = N / 6;
    for (int b = 0; b < nb; b++) {
        float* o0 = Out + (size_t)(b * 6) * M;
        int m = m0;
        for (; m + 7 < m1; m += 8) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            const float* xk = X + m;
            const float* wpk = Wp + b * 6;
            for (int k = 0; k < K; k++, xk += M, wpk += N) {
                __m256 xv = _mm256_loadu_ps(xk);
                // 6 ��ͨ��ֵ�������㲥����ͨ��
                acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 0), xv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 1), xv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 2), xv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 3), xv, acc3);
                acc4 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 4), xv, acc4);
                acc5 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 5), xv, acc5);
            }
            __m256 vones = _mm256_set1_ps(1.0f);
            __m256 vs[6] = {acc0, acc1, acc2, acc3, acc4, acc5};
            for (int i = 0; i < 6; i++) {
                __m256 v = vs[i];
                if (bias) v = _mm256_add_ps(v, _mm256_set1_ps(bias[b * 6 + i]));
                if (silu) {
                    __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), v);
                    t = _mm256_min_ps(t, _mm256_set1_ps(86.0f));
                    t = _mm256_max_ps(t, _mm256_set1_ps(-86.0f));
                    t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                    __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                    v = _mm256_div_ps(v, _mm256_add_ps(vones, ex));
                }
                _mm256_storeu_ps(o0 + (size_t)i * M + m, v);
            }
        }
        for (; m < m1; m++) {
            float s[6] = {0,0,0,0,0,0};
            for (int k = 0; k < K; k++) {
                float xv = X[(size_t)k * M + m];
                const float* wpk = Wp + (size_t)k * N + b * 6;
                for (int i = 0; i < 6; i++) s[i] += wpk[i] * xv;
            }
            for (int i = 0; i < 6; i++) {
                float v = s[i] + (bias ? bias[b * 6 + i] : 0.0f);
                if (silu) v = v / (1.0f + fast_exp(-v));
                o0[(size_t)i * M + m] = v;
            }
        }
    }
}

// 4-lane W packing for N%4==0 (256/512/1024 ch 1x1 convs; the 6-lane path
// only fires when N%6==0 so most YOLO 1x1 heads never got the packed GEMM).
struct PackedW4 { std::vector<float> data; int N = 0, K = 0; };
static std::unordered_map<const void*, PackedW4> g_packedW4;
static std::mutex g_packedW4_mtx;
static const float* get_packed_W4(const float* W, int N, int K) {
    if (N % 4 != 0 || K < 32) return nullptr;
    std::lock_guard<std::mutex> lk(g_packedW4_mtx);
    auto it = g_packedW4.find(W);
    if (it != g_packedW4.end() && it->second.N == N && it->second.K == K) return it->second.data.data();
    PackedW4 pw; pw.N = N; pw.K = K;
    pw.data.resize((size_t)K * N);
    int nb = N / 4;
    for (int k = 0; k < K; k++)
        for (int b = 0; b < nb; b++)
            for (int i = 0; i < 4; i++)
                pw.data[(size_t)k * N + b * 4 + i] = W[(b * 4 + i) * K + k];
    g_packedW4[W] = std::move(pw);
    return g_packedW4[W].data.data();
}

static inline void clear_packed_W4_cache() {
    std::lock_guard<std::mutex> lk(g_packedW4_mtx);
    g_packedW4.clear();
}

static inline void gemm_nm_packed4_core(const float* X, const float* Wp, const float* bias,
                                         float* Out, int N, int M, int K, bool silu,
                                         int m0, int m1) {
    int nb = N / 4;
    const __m256 vones = _mm256_set1_ps(1.0f);
    for (int b = 0; b < nb; b++) {
        float* o0 = Out + (size_t)(b * 4) * M;
        int m = m0;
        for (; m + 7 < m1; m += 8) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            const float* xk = X + m;
            const float* wpk = Wp + b * 4;
            for (int k = 0; k < K; k++, xk += M, wpk += N) {
                __m256 xv = _mm256_loadu_ps(xk);
                acc0 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 0), xv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 1), xv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 2), xv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_broadcast_ss(wpk + 3), xv, acc3);
            }
            __m256 vs[4] = {acc0, acc1, acc2, acc3};
            for (int i = 0; i < 4; i++) {
                __m256 v = vs[i];
                if (bias) v = _mm256_add_ps(v, _mm256_set1_ps(bias[b * 4 + i]));
                if (silu) {
                    __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), v);
                    t = _mm256_min_ps(t, _mm256_set1_ps(86.0f));
                    t = _mm256_max_ps(t, _mm256_set1_ps(-86.0f));
                    t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                    __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                    v = _mm256_div_ps(v, _mm256_add_ps(vones, ex));
                }
                _mm256_storeu_ps(o0 + (size_t)i * M + m, v);
            }
        }
        for (; m < m1; m++) {
            float s[4] = {0, 0, 0, 0};
            for (int k = 0; k < K; k++) {
                float xv = X[(size_t)k * M + m];
                const float* wpk = Wp + (size_t)k * N + b * 4;
                for (int i = 0; i < 4; i++) s[i] += wpk[i] * xv;
            }
            for (int i = 0; i < 4; i++) {
                float v = s[i] + (bias ? bias[b * 4 + i] : 0.0f);
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

    // pick packed lanes: 6 if N%6==0, else 4 if N%4==0 (covers 64/128/256/512/1024)
    int lanes = 0; // default keep legacy path (esp. x86, where 4-lane pack regressed ~4x)
#if defined(_WIN64) && defined(__AVX2__)
    lanes = (N % 6 == 0) ? 6 : ((N % 4 == 0) ? 4 : 0);
#endif
    const float* Wp = nullptr;
    if (lanes == 6 && K >= 32) Wp = get_packed_W(W, N, K);
    else if (lanes == 4 && K >= 32) Wp = get_packed_W4(W, N, K);

    size_t xbytes = (size_t)K * M * sizeof(float);
    int CH = lanes ? lanes : 6;
    if (!Wp && xbytes <= (2u << 20)) {
        while (CH > 1 && (N + CH - 1) / CH < workers * 2) CH--;
    }
    int nb = (N + CH - 1) / CH;
    int mb = 1;
    while (nb * mb < workers * 2 && mb < 8 && (M / (mb * 2)) >= 64) mb *= 2;
    int m_chunk = ((M + mb - 1) / mb + 7) & ~7;
    int m_blocks = (M + m_chunk - 1) / m_chunk;
    int tasks = nb * m_blocks;
    if (tasks <= 1) {
        if (Wp) {
            for (int b = 0; b < nb; b++) {
                int n0 = b * lanes, n1 = std::min(N, n0 + lanes);
                if (n1 - n0 == lanes) {
                    if (lanes == 4) gemm_nm_packed4_core(X, Wp, bias ? bias + n0 : nullptr, Out, N, M, K, silu, 0, M);
                    else gemm_nm_packed_core(X, Wp, bias ? bias + n0 : nullptr, Out, N, M, K, silu, 0, M);
                } else {
                    gemm_nm_core(X, W + (size_t)n0 * K, bias ? bias + n0 : nullptr, Out + (size_t)n0 * M, n1 - n0, M, K, silu, 0, M);
                }
            }
        } else {
            gemm_nm_core(X, W, bias, Out, N, M, K, silu);
        }
        return;
    }

    if (Wp) {
        SimpleThreadPool::instance().parallel_for(m_blocks, [&](int mk) {
            int ma = mk * m_chunk;
            int mz = std::min(M, ma + m_chunk);
            if (lanes == 4) gemm_nm_packed4_core(X, Wp, bias, Out, N, M, K, silu, ma, mz);
            else           gemm_nm_packed_core(X, Wp, bias, Out, N, M, K, silu, ma, mz);
        });
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

// �ֲ߳̾���ʱ������������ col ���м���㣬����Ƶ��?malloc/free�����̰߳�ȫ��
static inline float* get_temp_buf(size_t numel) {
    struct Holder {
        float* p = nullptr;
        size_t cap = 0;
        ~Holder() { if (p) free(p); }
    };
    static thread_local Holder buf;
    if (numel > buf.cap) {
        float* np = (float*)realloc(buf.p, numel * sizeof(float));
        if (!np) {
            // 这里原先返回 nullptr，而 6 个调用点全都不判空，下一句就是
            // `ws + offset` / memset —— Winograd 大层单次申请可达 ~70MB，
            // 32 位下失败即空指针解引用崩溃。改为抛异常，由 run() 捕获转错误码。
            throw std::bad_alloc();
        }
        buf.p = np;
        buf.cap = numel;
    }
    return buf.p;
}

// 3x3 s1 p1 ר�þ��� v2���ο� ncnn��ͬʱ��������������м��й�����?
// AVX2 �Ż���ÿ��һ�δ��� 8 ������
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

        // 2. ��ÿ�����ͨ�������̣߳������ͨ���鲢�У�
        const int P_BLOCK = 4;
        int num_blocks = (C_out + P_BLOCK - 1) / P_BLOCK;
        SimpleThreadPool::instance().parallel_for(num_blocks, [&](int block_idx) {
            int p_begin = block_idx * P_BLOCK;
            int p_end = std::min(C_out, p_begin + P_BLOCK);
            for (int p = p_begin; p < p_end; p++) {
            float* out_p = out_n + (size_t)p * outH * outW;
            float b = bias ? bias[p] : 0.0f;
            for (int i = 0; i < outH * outW; i++) out_p[i] = b;

        // 3. ��ÿ������ͨ���ۼ�
            for (int q = 0; q < C_in; q++) {
                const float* img = padded + (size_t)q * pH * pW;
                const float* k = weight + ((size_t)p * C_in + q) * 9;
                const float* k0 = k;
                const float* k1 = k + 3;
                const float* k2 = k + 6;

                // Ԥ����Ȩ�ص� AVX2 �Ĵ���
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
                // ͬʱ�����������?
                for (; h + 1 < outH; h += 2) {
                    float* outptr2 = outptr + outW;

                    int w = 0;
                    // AVX2+FMA��ÿ��һ�δ��� 8 ������
                    for (; w + 7 < outW; w += 8) {
                        // ��һ�������r0, r1, r2
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

                        // �ڶ��������r1, r2, r3��r1, r2 ���һ�й�����?
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
                    // ʣ�����أ�������
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

                    // ������һ����
                    r0 += 2 * pW;
                    r1 += 2 * pW;
                    r2 += 2 * pW;
                    r3 += 2 * pW;
                    outptr += 2 * outW;
                }
                // ����ʣ��ĵ���?
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

// 3x3 s2 p1 ֱ�Ӿ�����ʡ im2col��ֱ�Ӽ��㣬AVX2 + ���У�
static inline void conv3x3s2p1(const float* input, const float* weight, const float* bias,
                               float* output, int N, int C_in, int H, int W, int C_out, bool silu) {
    int outH = (H + 2 - 3) / 2 + 1;
    int outW = (W + 2 - 3) / 2 + 1;
    int pH = H + 2, pW = W + 2;
    size_t pad_numel = (size_t)C_in * pH * pW;
    float* padded = get_temp_buf(pad_numel);
    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C_in * H * W;
        float* out_n = output + (size_t)n * C_out * outH * outW;
        // padding
        memset(padded, 0, pad_numel * sizeof(float));
        for (int c = 0; c < C_in; c++) {
            const float* in_c = in_n + (size_t)c * H * W;
            float* pad_c = padded + (size_t)c * pH * pW + pW + 1;
            for (int h = 0; h < H; h++) memcpy(pad_c + h * pW, in_c + h * W, W * sizeof(float));
        }
        const int P_BLOCK = 4;
        int num_blocks = (C_out + P_BLOCK - 1) / P_BLOCK;
        SimpleThreadPool::instance().parallel_for(num_blocks, [&](int block_idx) {
            int p_begin = block_idx * P_BLOCK;
            int p_end = std::min(C_out, p_begin + P_BLOCK);
            for (int p = p_begin; p < p_end; p++) {
                float* out_p = out_n + (size_t)p * outH * outW;
                float b = bias ? bias[p] : 0.0f;
                for (int i = 0; i < outH * outW; i++) out_p[i] = b;
                for (int q = 0; q < C_in; q++) {
                    const float* img = padded + (size_t)q * pH * pW;
                    const float* k = weight + ((size_t)p * C_in + q) * 9;
                    for (int oh = 0; oh < outH; oh++) {
                        const float* r0 = img + (oh * 2) * pW;
                        const float* r1 = img + (oh * 2 + 1) * pW;
                        const float* r2 = img + (oh * 2 + 2) * pW;
                        float* outptr = out_p + oh * outW;
                        int ow = 0;
                        // stride=2 ʱ��������Ĳ��������?2�����ܰ� s1 ��ʽ���� 8 �����ء�
                        // ��ȷ������ÿ 4 �����һ�飬���ڰ�?+0/+2/+4/+6 ƫ�Ƹ���һ��
                        // 4 �����أ��� 3 �� lane ǡ���Ǹ������?3 ������ tap��
                        __m128 vk01 = _mm_set_ps(0.0f, k[2], k[1], k[0]);
                        __m128 vk34 = _mm_set_ps(0.0f, k[5], k[4], k[3]);
                        __m128 vk67 = _mm_set_ps(0.0f, k[8], k[7], k[6]);
                        for (; ow + 3 < outW; ow += 4) {
                            int iw = ow * 2;
                            // �� 1 ������ (k0,k1,k2)
                            __m128 a0 = _mm_mul_ps(_mm_loadu_ps(r0 + iw), vk01);
                            __m128 a1 = _mm_mul_ps(_mm_loadu_ps(r0 + iw + 2), vk01);
                            __m128 a2 = _mm_mul_ps(_mm_loadu_ps(r0 + iw + 4), vk01);
                            __m128 a3 = _mm_mul_ps(_mm_loadu_ps(r0 + iw + 6), vk01);
                            // �� 2 ������ (k3,k4,k5)
                            a0 = _mm_fmadd_ps(_mm_loadu_ps(r1 + iw), vk34, a0);
                            a1 = _mm_fmadd_ps(_mm_loadu_ps(r1 + iw + 2), vk34, a1);
                            a2 = _mm_fmadd_ps(_mm_loadu_ps(r1 + iw + 4), vk34, a2);
                            a3 = _mm_fmadd_ps(_mm_loadu_ps(r1 + iw + 6), vk34, a3);
                            // �� 3 ������ (k6,k7,k8)
                            a0 = _mm_fmadd_ps(_mm_loadu_ps(r2 + iw), vk67, a0);
                            a1 = _mm_fmadd_ps(_mm_loadu_ps(r2 + iw + 2), vk67, a1);
                            a2 = _mm_fmadd_ps(_mm_loadu_ps(r2 + iw + 4), vk67, a2);
                            a3 = _mm_fmadd_ps(_mm_loadu_ps(r2 + iw + 6), vk67, a3);
                            // ˮƽ��ͣ�lane3 ��Ϊ 0��v0+v1+v2 �����?
                            __m128 t0 = _mm_add_ps(a0, _mm_movehl_ps(a0, a0));
                            __m128 t1 = _mm_add_ps(a1, _mm_movehl_ps(a1, a1));
                            __m128 t2 = _mm_add_ps(a2, _mm_movehl_ps(a2, a2));
                            __m128 t3 = _mm_add_ps(a3, _mm_movehl_ps(a3, a3));
                            t0 = _mm_add_ss(t0, _mm_shuffle_ps(t0, t0, _MM_SHUFFLE(1, 1, 1, 1)));
                            t1 = _mm_add_ss(t1, _mm_shuffle_ps(t1, t1, _MM_SHUFFLE(1, 1, 1, 1)));
                            t2 = _mm_add_ss(t2, _mm_shuffle_ps(t2, t2, _MM_SHUFFLE(1, 1, 1, 1)));
                            t3 = _mm_add_ss(t3, _mm_shuffle_ps(t3, t3, _MM_SHUFFLE(1, 1, 1, 1)));
                            outptr[ow + 0] += _mm_cvtss_f32(t0);
                            outptr[ow + 1] += _mm_cvtss_f32(t1);
                            outptr[ow + 2] += _mm_cvtss_f32(t2);
                            outptr[ow + 3] += _mm_cvtss_f32(t3);
                        }
                        for (; ow < outW; ow++) {
                            int iw2 = ow * 2;
                            outptr[ow] += r0[iw2]*k[0] + r0[iw2+1]*k[1] + r0[iw2+2]*k[2]
                                        + r1[iw2]*k[3] + r1[iw2+1]*k[4] + r1[iw2+2]*k[5]
                                        + r2[iw2]*k[6] + r2[iw2+1]*k[7] + r2[iw2+2]*k[8];
                        }
                    }
                }
                // SiLU ��������������ͨ���ۼ���ɺ�ͳһӦ��?
                // ����ʵ���� q ѭ���ڲ���ͨ��Ӧ�ã��ȼ��ڶԲ��ֺ���������������˸�·����ǰ�����ã�
                if (silu) {
                    int cnt = outH * outW;
                    int i = 0;
                    const __m256 vones = _mm256_set1_ps(1.0f);
                    for (; i + 7 < cnt; i += 8) {
                        __m256 v = _mm256_loadu_ps(out_p + i);
                        __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), v);
                    t = _mm256_min_ps(t, _mm256_set1_ps(86.0f));
                    t = _mm256_max_ps(t, _mm256_set1_ps(-86.0f));
                        t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                        __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                        v = _mm256_div_ps(v, _mm256_add_ps(vones, ex));
                        _mm256_storeu_ps(out_p + i, v);
                    }
                    for (; i < cnt; i++) {
                        float x = out_p[i];
                        out_p[i] = x / (1.0f + fast_exp(-x));
                    }
                }
            }
        });
    }
}

// ============================================================
// Winograd F(2,3) ������3x3 s1 p1������ 2.25x �˷���
// �任���Ȩ�ػ��棨Ȩ��ָ��?-> U������֡���ã�����ÿ֡����
// ============================================================
struct WinogradCacheEntry {
    std::vector<float> u;
    int kt = 0;   // �任�� tile �� k ά��С��F(2,3)=16��F(4,3)=36��������ͬȨ�صĲ�ͬ�任
};
struct WinogradWeightCache {
    std::unordered_map<const void*, WinogradCacheEntry> map;
    std::mutex mtx;
    void clear() {
        std::lock_guard<std::mutex> lock(mtx);
        map.clear();
    }
};
static WinogradWeightCache g_winograd_cache;
static inline void clear_winograd_weight_cache() { g_winograd_cache.clear(); }

// ǰ���������������·���
static inline void winograd_weight_transform(const float* g, float* U);
static inline void winograd_input_transform(const float* d, float* V);
static inline void winograd_output_transform(const float* M, float* Y);
static inline void wino43_weight_transform(const float* g, float* U);
static inline void wino43_input_transform(const float* d, float* V);
static inline void wino43_output_transform(const float* M, float* Y);

// ����GEMM�汾����Winogradת����16��СGEMM��ֱ���� gemm_nm_core
// M[k][C_out, T] = U[k][C_out, C_in] * V[k][C_in, T]��V ��Ȼ�� [K=C_in][M=T] ���֣�����ת��
// TEMP profiling
static double g_stage_pad = 0, g_stage_vt = 0, g_stage_gemm = 0, g_stage_ot = 0;
static int g_stage_n = 0;
static inline double now_ms_() {
    return (double)GetTickCount64();
}
// ģ�����?TS = tile ����߳���? => F(2,3)��4 => F(4,3)����
// ��֤ D/KT �Ǳ����ڳ�������ѭ��������չ��
// ---------------------------------------------------------------------------
// SIMD batched F(4,3) transforms: 8 tiles per AVX2 lane-group.
// Input : V = BT*D*BT^T (6x6 window -> 6x6), loads via AVX2 gather from the
//         padded plane; stores into the existing [T][KT] scratch (strided).
// Output: Y = AT*M*AT^T (6x6 -> 4x4), Mbuf rows continuous per k -> loadu.
// Both transforms reuse one coefficient helper per stage.
// ---------------------------------------------------------------------------

// o[i] = BT[i] dot x  (BT = [[4,0,-5,0,1,0],[0,-4,-4,1,1,0],[0,4,-4,-1,1,0],
//                             [0,-2,-1,2,1,0],[0,2,-1,-2,1,0],[0,4,0,-5,0,1]])
static inline void f4_bt6_row(const __m256 x[6], __m256 o[6]) {
    o[0] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), x[0], _mm256_fmadd_ps(_mm256_set1_ps(-5.0f), x[2], x[4]));
    o[1] = _mm256_fmadd_ps(_mm256_set1_ps(-4.0f), x[1], _mm256_fmadd_ps(_mm256_set1_ps(-4.0f), x[2], _mm256_add_ps(x[3], x[4])));
    o[2] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), x[1], _mm256_fmadd_ps(_mm256_set1_ps(-4.0f), x[2], _mm256_sub_ps(x[4], x[3])));
    o[3] = _mm256_fmadd_ps(_mm256_set1_ps(-2.0f), x[1], _mm256_fmadd_ps(_mm256_set1_ps(-1.0f), x[2], _mm256_fmadd_ps(_mm256_set1_ps(2.0f), x[3], x[4])));
    o[4] = _mm256_fmadd_ps(_mm256_set1_ps(2.0f), x[1], _mm256_fmadd_ps(_mm256_set1_ps(-1.0f), x[2], _mm256_fmadd_ps(_mm256_set1_ps(-2.0f), x[3], x[4])));
    o[5] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), x[1], _mm256_fmadd_ps(_mm256_set1_ps(-5.0f), x[3], x[5]));
}

// o[r] = AT[r] dot x  (AT = [[1,1,1,1,1,0],[0,1,-1,2,-2,0],[0,1,1,4,4,0],[0,1,-1,8,-8,1]])
static inline void f4_at4_row(const __m256 x[6], __m256 o[4]) {
    o[0] = _mm256_add_ps(_mm256_add_ps(x[0], x[1]), _mm256_add_ps(_mm256_add_ps(x[2], x[3]), x[4]));
    o[1] = _mm256_fmadd_ps(_mm256_set1_ps(2.0f), x[3], _mm256_fmadd_ps(_mm256_set1_ps(-2.0f), x[4], _mm256_sub_ps(x[1], x[2])));
    o[2] = _mm256_fmadd_ps(_mm256_set1_ps(4.0f), x[3], _mm256_fmadd_ps(_mm256_set1_ps(4.0f), x[4], _mm256_add_ps(x[1], x[2])));
    o[3] = _mm256_fmadd_ps(_mm256_set1_ps(8.0f), x[3], _mm256_fmadd_ps(_mm256_set1_ps(-8.0f), x[4], _mm256_fmadd_ps(_mm256_set1_ps(1.0f), x[5], _mm256_sub_ps(x[1], x[2]))));
}

// input transform for 8 consecutive tiles t0..t0+7; dst is the [T][KT] scratch.
static inline void wino43_input_transform8(const float* padded, float* dst,
                                           int T, int w_tiles, int pW, int t0) {
    __m256i idx;
    {
        int off[8];
        for (int i = 0; i < 8; i++) {
            int tt = t0 + i;
            int ht = tt / w_tiles, wt = tt % w_tiles;
            off[i] = (ht * 4) * pW + wt * 4; // element offsets from channel base
        }
        idx = _mm256_setr_epi32(off[0], off[1], off[2], off[3], off[4], off[5], off[6], off[7]);
    }
    __m256 d[6][6], tmp[6][6], v[6][6];
    for (int j = 0; j < 6; j++)
        for (int c = 0; c < 6; c++)
            d[j][c] = _mm256_i32gather_ps(padded + j * pW + c, idx, 4);
    // tmp[i][c] = BT[i] dot d[.][c]
    for (int c = 0; c < 6; c++) {
        __m256 x[6], o[6];
        for (int j = 0; j < 6; j++) x[j] = d[j][c];
        f4_bt6_row(x, o);
        for (int i = 0; i < 6; i++) tmp[i][c] = o[i];
    }
    // v[i][s] = BT[s] dot tmp[i][.]
    for (int i = 0; i < 6; i++) f4_bt6_row(tmp[i], v[i]);
    float tv[8];
    for (int kk = 0; kk < 36; kk++) {
        _mm256_storeu_ps(tv, v[kk / 6][kk % 6]);
        for (int i = 0; i < 8; i++) dst[(size_t)(t0 + i) * 36 + kk] = tv[i];
    }
}

// output transform for 8 consecutive tiles t0..t0+7 (bias + fused SiLU folded in).
static inline void wino43_output_transform8(const float* Mbuf, float* out_n, const float* bias, bool silu,
                                            int C_out, int p, int T, int w_tiles, int outH, int outW, int t0) {
    __m256 m[36], tmp[4][6], y[4][4];
    for (int k = 0; k < 36; k++)
        m[k] = _mm256_loadu_ps(Mbuf + ((size_t)k * C_out + p) * T + t0);
    for (int c = 0; c < 6; c++) {
        __m256 x[6], o[4];
        for (int j = 0; j < 6; j++) x[j] = m[j * 6 + c];
        f4_at4_row(x, o);
        for (int r = 0; r < 4; r++) tmp[r][c] = o[r];
    }
    for (int r = 0; r < 4; r++) f4_at4_row(tmp[r], y[r]);
    const __m256 vb = _mm256_set1_ps(bias ? bias[p] : 0.0f);
    for (int r = 0; r < 4; r++)
        for (int s = 0; s < 4; s++) {
            __m256 vv = _mm256_add_ps(y[r][s], vb);
            if (silu) {
                __m256 tv = _mm256_sub_ps(_mm256_setzero_ps(), vv);
                tv = _mm256_min_ps(tv, _mm256_set1_ps(86.0f));
                tv = _mm256_max_ps(tv, _mm256_set1_ps(-86.0f));
                tv = _mm256_fmadd_ps(tv, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(tv));
                vv = _mm256_div_ps(vv, _mm256_add_ps(_mm256_set1_ps(1.0f), ex));
            }
            y[r][s] = vv;
        }
    float yv[4][4][8];
    for (int r = 0; r < 4; r++)
        for (int s = 0; s < 4; s++)
            _mm256_storeu_ps(yv[r][s], y[r][s]);
    for (int i = 0; i < 8; i++) {
        int tt = t0 + i;
        int ht = tt / w_tiles, wt = tt % w_tiles;
        int oh = ht * 4, ow = wt * 4;
        float* op = out_n + (size_t)p * outH * outW;
        for (int r = 0; r < 4 && oh + r < outH; r++)
            for (int s = 0; s < 4 && ow + s < outW; s++)
                op[(size_t)(oh + r) * outW + (ow + s)] = yv[r][s][i];
    }
}

template<int TS>
static inline void conv3x3s1_wino_tmpl(const float* input, const float* weight, const float* bias,
                                        float* output, bool apply_silu,
                                        int N, int C_in, int H, int W, int C_out) {
    static const bool g_no_simd = (getenv("TINY_YOLO_NO_SIMD") != nullptr); // A/B: disable F4 SIMD transforms
    constexpr int D  = TS + 2;   // ���봰�ڱ߳����� kernel ���ǣ�
    constexpr int KT = D * D;    // k �ռ��С��F(2,3)=16��F(4,3)=36��

    int outH = H, outW = W;
    // padding �踲�����һ��?tile ���������ڣ������� = h_tiles*TS + 1
    int h_tiles = (outH + TS - 1) / TS;
    int w_tiles = (outW + TS - 1) / TS;
    int pH = h_tiles * TS + 2, pW = w_tiles * TS + 2;

    // tile����
    int T = h_tiles * w_tiles;

    // ���� padded ���루�����һ�У���ֹ�߽�?tile Խ�����?
    // ע�⣺get_temp_buf �ǵ����ֲ߳̾����壨realloc ���ƶ�ָ�룩��
    // ����һ��������ȫ��������������ƫ���зֳ���������
    size_t pad_numel = (size_t)C_in * pH * pW;

    // �м仺������С��U ֱ�����û��棬���ٿ��빤������
    size_t V_size = (size_t)KT * C_in * T;
    size_t M_size = (size_t)KT * C_out * T;
    // dense transform scratch only when per-conv size is small; for 640-class
    // maps a full C_in*KT*T scratch would be tens of MB and hurt cache
    const bool vt_scratch = ((size_t)C_in * (size_t)KT * (size_t)T) <= ((size_t)2 << 20);
    size_t total = pad_numel + pW + V_size + M_size + (vt_scratch ? (size_t)C_in * (size_t)KT * (size_t)T : 0);
    float* ws = get_temp_buf(total);
    float* padded = ws;
    float* V = ws + pad_numel + pW;
    float* Mbuf = V + V_size;

    // 1. Ȩ�ر任 U[KT, C_out, C_in] = G * g * G^T�������棬����ֱ�����û���ָ�룩
    const float* U = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_winograd_cache.mtx);
        auto it = g_winograd_cache.map.find(weight);
        if (it != g_winograd_cache.map.end() && it->second.kt == KT) {
            U = it->second.u.data();
        } else {
            WinogradCacheEntry entry;
            entry.kt = KT;
            entry.u.resize((size_t)KT * C_out * C_in);
            for (int p = 0; p < C_out; p++) {
                for (int q = 0; q < C_in; q++) {
                    const float* g = weight + ((size_t)p * C_in + q) * 9;
                    float u[36];
                    if (TS == 4) wino43_weight_transform(g, u);
                    else         winograd_weight_transform(g, u);
                    for (int k = 0; k < KT; k++) {
                        entry.u[((size_t)k * C_out + p) * C_in + q] = u[k];
                    }
                }
            }
            auto res = g_winograd_cache.map.insert_or_assign(weight, std::move(entry));
            U = res.first->second.u.data();
        }
    }

    for (int n = 0; n < N; n++) {
        const float* in_n = input + (size_t)n * C_in * H * W;
        float* out_n = output + (size_t)n * C_out * outH * outW;

        // 2. �������� padding��������ͨ�����У���ͨ���ڴ����������?
        {
            SimpleThreadPool& pool_ = SimpleThreadPool::instance();
            bool pmt = (pool_.num_threads() > 1 && pad_numel >= 32768);
            auto pad_body = [&](int c) {
                const float* in_c = in_n + (size_t)c * H * W;
                float* pad_c = padded + (size_t)c * pH * pW + pW + 1;
                float* pad_row0 = padded + (size_t)c * pH * pW;
                // ��ͨ���������߽���/������������ memset ��ɣ�����ֻ�追������?
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
        // One channel may be processed by either path: dense scratch makes the
        // k-plane stores contiguous (better for small 256-class maps), while the
        // direct path keeps memory flat for large 640-class maps.
        auto vt_channel = [&](int q) {
            const float* img_q = padded + (size_t)q * pH * pW;
            if (vt_scratch) {
                float* scr = ws + pad_numel + pW + V_size + M_size + (size_t)q * (size_t)KT * T;
                if (TS == 4 && T >= 8 && !g_no_simd) {
                    // SIMD: F(4,3) gathers 8 tiles per batch
                    int t = 0;
                    for (; t + 8 <= T; t += 8) wino43_input_transform8(img_q, scr, T, w_tiles, pW, t);
                    for (; t < T; t++) {
                        float v[36];
                        int ht = t / w_tiles;
                        int wt = t % w_tiles;
                        int oh = ht * TS, ow = wt * TS;
                        const float* d_src = img_q + oh * pW + ow;
                        float d_block[36];
                        for (int r = 0; r < D; r++)
                            memcpy(d_block + r * D, d_src + r * pW, D * sizeof(float));
                        wino43_input_transform(d_block, v);
                        for (int kk = 0; kk < KT; kk++) scr[(size_t)t * KT + kk] = v[kk];
                    }
                } else {
                for (int t = 0; t < T; t++) {
                    float v[36];
                    int ht = t / w_tiles;
                    int wt = t % w_tiles;
                    int oh = ht * TS, ow = wt * TS;
                    const float* d_src = img_q + oh * pW + ow;
                    float d_block[36];
                    for (int r = 0; r < D; r++)
                        memcpy(d_block + r * D, d_src + r * pW, D * sizeof(float));
                    if (TS == 4) wino43_input_transform(d_block, v);
                    else         winograd_input_transform(d_block, v);
                    for (int kk = 0; kk < KT; kk++) scr[(size_t)t * KT + kk] = v[kk];
                }
                }
                float* v_q = V + (size_t)q * T;
                for (int k = 0; k < KT; k++) {
                    float* plane = v_q + (size_t)k * (size_t)C_in * T;
                    for (int t = 0; t < T; t++) plane[t] = scr[(size_t)t * KT + k];
                }
            } else {
                float* v_q = V + (size_t)q * T;
                for (int t = 0; t < T; t++) {
                    float v[36];
                    int ht = t / w_tiles;
                    int wt = t % w_tiles;
                    int oh = ht * TS, ow = wt * TS;
                    const float* d_src = img_q + oh * pW + ow;
                    float d_block[36];
                    for (int r = 0; r < D; r++)
                        memcpy(d_block + r * D, d_src + r * pW, D * sizeof(float));
                    if (TS == 4) wino43_input_transform(d_block, v);
                    else         winograd_input_transform(d_block, v);
                    float* dst = v_q;
                    for (int k = 0; k < KT; k++, dst += (size_t)C_in * T) dst[t] = v[k];
                }
            }
        };
        if ((long long)C_in * T > 8192 && C_in >= 2) {
            const int QC = 4;
            int qblocks = (C_in + QC - 1) / QC;
            SimpleThreadPool::instance().parallel_for(qblocks, [&](int qb) {
                int q_begin = qb * QC;
                int q_end = std::min(C_in, q_begin + QC);
                for (int q = q_begin; q < q_end; q++) vt_channel(q);
            });
        } else {
            for (int q = 0; q < C_in; q++) vt_channel(q);
        }

        // 4. KT ��С GEMM��M[k] = U[k] �� V[k]���� k ���У��ڲ����б���Ƕ�ײ��У�
        SimpleThreadPool& pool = SimpleThreadPool::instance();
        bool gemm_mt = ((long long)KT * C_out * T * C_in > 500000 && pool.num_threads() > 1);
        if (gemm_mt) {
            pool.parallel_for(KT, [&](int k) {
                gemm_nm_core(V + (size_t)k * C_in * T,
                             U + (size_t)k * C_out * C_in,
                             nullptr,
                             Mbuf + (size_t)k * C_out * T,
                             C_out, T, C_in, false);
            });
        } else {
            for (int k = 0; k < KT; k++) {
                gemm_nm_core(V + (size_t)k * C_in * T,
                             U + (size_t)k * C_out * C_in,
                             nullptr,
                             Mbuf + (size_t)k * C_out * T,
                             C_out, T, C_in, false);
            }
        }

        // 5. �����?+ bias + SiLU + д�أ������ͨ���ֿ鲢�У�?
        const int PC = 8;
        int pblocks = (C_out + PC - 1) / PC;
        bool out_mt = ((long long)C_out * T > 8192 && pool.num_threads() > 1);
        // scalar transform for one (p, t); shared by SIMD tail and TS=2 path
        auto ot_tile = [&](int p, int t) {
            float m[36], y[16];
            int ht = t / w_tiles;
            int wt = t % w_tiles;
            int oh = ht * TS, ow = wt * TS;
            for (int k = 0; k < KT; k++) m[k] = Mbuf[((size_t)k * C_out + p) * T + t];
            if (TS == 4) wino43_output_transform(m, y);
            else         winograd_output_transform(m, y);
            float b = bias ? bias[p] : 0.0f;
            for (int i = 0; i < TS * TS; i++) {
                float val = y[i] + b;
                if (apply_silu) val = val / (1.0f + fast_exp(-val));
                y[i] = val;
            }
            float* out_p = out_n + (size_t)p * outH * outW;
            for (int dh = 0; dh < TS && oh + dh < outH; dh++)
                for (int dw = 0; dw < TS && ow + dw < outW; dw++)
                    out_p[(oh + dh) * outW + (ow + dw)] = y[dh * TS + dw];
        };
        auto out_body = [&](int pb) {
            int p_begin = pb * PC;
            int p_end = std::min(C_out, p_begin + PC);
            if (TS == 4 && T >= 8 && !g_no_simd) {
                for (int p = p_begin; p < p_end; p++) {
                    int t = 0;
                    for (; t + 8 <= T; t += 8) wino43_output_transform8(Mbuf, out_n, bias, apply_silu, C_out, p, T, w_tiles, outH, outW, t);
                    for (; t < T; t++) ot_tile(p, t);
                }
            } else {
                for (int t = 0; t < T; t++)
                    for (int p = p_begin; p < p_end; p++) ot_tile(p, t);
            }
        };
        if (out_mt && pblocks >= 2) {
            pool.parallel_for(pblocks, out_body);
        } else {
            for (int pb = 0; pb < pblocks; pb++) out_body(pb);
        }
    }
}

// Winograd ��ڣ�mode 0=�Զ�����ͨ����ѡ F(2,3)/F(4,3)����1=ǿ�� F(2,3)��2=ǿ�� F(4,3)
// F(4,3) �� GEMM �������� F(2,3) �� 56%�����任����������
// ���� C_in*C_out �㹻�󣨱任ռ�ȿɺ��ԣ�ʱ����
static inline void conv3x3s1_winograd(const float* input, const float* weight, const float* bias,
                                      float* output, bool apply_silu,
                                      int N, int C_in, int H, int W, int C_out,
                                      int mode = 0) {
    bool f4 = (mode == 2) || (mode == 0 && (long long)C_in * C_out >= 2048);
    if (f4) conv3x3s1_wino_tmpl<4>(input, weight, bias, output, apply_silu, N, C_in, H, W, C_out);
    else    conv3x3s1_wino_tmpl<2>(input, weight, bias, output, apply_silu, N, C_in, H, W, C_out);
}

// Ȩ�ر任��U = G * g * G^T��g �� 3x3��U �� 4x4
static inline void winograd_weight_transform(const float* g, float* U) {
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    float tmp[4][3];
    // tmp = G * g������ j ѭ����tmp[i][j] = sum_k G[i][k] * g[k][j]
    for (int j = 0; j < 3; j++) {
        float g0 = g[0*3+j], g1 = g[1*3+j], g2 = g[2*3+j];
        tmp[0][j] = g0;
        tmp[1][j] = 0.5f * (g0 + g1 + g2);
        tmp[2][j] = 0.5f * (g0 - g1 + g2);
        tmp[3][j] = g2;
    }
    // U = tmp * G^T������ i ѭ����U[i][j] = sum_k tmp[i][k] * G[j][k]
    for (int i = 0; i < 4; i++) {
        float t0 = tmp[i][0], t1 = tmp[i][1], t2 = tmp[i][2];
        U[i*4+0] = t0;
        U[i*4+1] = 0.5f * (t0 + t1 + t2);
        U[i*4+2] = 0.5f * (t0 - t1 + t2);
        U[i*4+3] = t2;
    }
}

// ����任��V = BT * d * BT^T��d �� 4x4��V �� 4x4
static inline void winograd_input_transform(const float* d, float* V) {
    // BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    float tmp[4][4];
    // �б任��tmp = BT * d
    for (int c = 0; c < 4; c++) {
        float d0 = d[0*4+c], d1 = d[1*4+c], d2 = d[2*4+c], d3 = d[3*4+c];
        tmp[0][c] = d0 - d2;
        tmp[1][c] = d1 + d2;
        tmp[2][c] = -d1 + d2;
        tmp[3][c] = d1 - d3;  // ������BT��4����[0,1,0,-1]�������� d1 - d3
    }
    // �б任��V = tmp * BT^T
    for (int r = 0; r < 4; r++) {
        float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2], t3 = tmp[r][3];
        V[r*4+0] = t0 - t2;
        V[r*4+1] = t1 + t2;
        V[r*4+2] = -t1 + t2;
        V[r*4+3] = t1 - t3;  // ������BT^T�ĵ�4����[0,1,0,-1]^T
    }
}

// ����任��Y = AT * M * AT^T��M �� 4x4��Y �� 2x2
static inline void winograd_output_transform(const float* M, float* Y) {
    // AT = [[1,1,1,0],[0,1,-1,-1]]
    float tmp[2][4];
    // �б任��tmp = AT * M
    for (int c = 0; c < 4; c++) {
        float m0 = M[0*4+c], m1 = M[1*4+c], m2 = M[2*4+c], m3 = M[3*4+c];
        tmp[0][c] = m0 + m1 + m2;
        tmp[1][c] = m1 - m2 - m3;
    }
    // �б任��Y = tmp * AT^T
    for (int r = 0; r < 2; r++) {
        float t0 = tmp[r][0], t1 = tmp[r][1], t2 = tmp[r][2], t3 = tmp[r][3];
        Y[r*2+0] = t0 + t1 + t2;
        Y[r*2+1] = t1 - t2 - t3;
    }
}

// ============================================================
// Winograd F(4,3)��6x6 tile ���?4x4���˷� 36/16=2.25 ÿ���?
// ��F(2,3) Ϊ 4 ÿ�������GEMM �׶μ��������� 56%
// ����ȡ�� Lavin & Gray (2016)�����з���ϵ������Ȩ�ر任 G �У����ߣ���
// ����ʱ������/�����?BT/AT ��Ϊ����ϵ��
// ============================================================
static inline void wino43_weight_transform(const float* g, float* U /*36*/) {
    static const float G[6][3] = {
        { 1.0f/4,   0.f,     0.f   },
        {-1.0f/6, -1.0f/6, -1.0f/6},
        {-1.0f/6,  1.0f/6, -1.0f/6},
        { 1.0f/24, 1.0f/12, 1.0f/6},
        { 1.0f/24,-1.0f/12, 1.0f/6},
        { 0.f,     0.f,     1.f   }};
    float tmp[6][3];
    for (int j = 0; j < 3; j++) {
        float g0 = g[0*3+j], g1 = g[1*3+j], g2 = g[2*3+j];
        for (int i = 0; i < 6; i++)
            tmp[i][j] = G[i][0]*g0 + G[i][1]*g1 + G[i][2]*g2;
    }
    for (int i = 0; i < 6; i++)
        for (int k = 0; k < 6; k++)
            U[i*6+k] = tmp[i][0]*G[k][0] + tmp[i][1]*G[k][1] + tmp[i][2]*G[k][2];
}

static inline void wino43_input_transform(const float* d /*36*/, float* V /*36*/) {
    static const float BT[6][6] = {
        { 4,  0, -5,  0,  1,  0},
        { 0, -4, -4,  1,  1,  0},
        { 0,  4, -4, -1,  1,  0},
        { 0, -2, -1,  2,  1,  0},
        { 0,  2, -1, -2,  1,  0},
        { 0,  4,  0, -5,  0,  1}};
    float tmp[6][6];
    // tmp = BT * d
    for (int c = 0; c < 6; c++) {
        float d0=d[0*6+c], d1=d[1*6+c], d2=d[2*6+c], d3=d[3*6+c], d4=d[4*6+c], d5=d[5*6+c];
        for (int i = 0; i < 6; i++)
            tmp[i][c] = BT[i][0]*d0 + BT[i][1]*d1 + BT[i][2]*d2 + BT[i][3]*d3 + BT[i][4]*d4 + BT[i][5]*d5;
    }
    // V = tmp * BT^T
    for (int i = 0; i < 6; i++) {
        float t0=tmp[i][0], t1=tmp[i][1], t2=tmp[i][2], t3=tmp[i][3], t4=tmp[i][4], t5=tmp[i][5];
        for (int j = 0; j < 6; j++)
            V[i*6+j] = BT[j][0]*t0 + BT[j][1]*t1 + BT[j][2]*t2 + BT[j][3]*t3 + BT[j][4]*t4 + BT[j][5]*t5;
    }
}

static inline void wino43_output_transform(const float* M /*36*/, float* Y /*16*/) {
    static const float AT[4][6] = {
        {1, 1, 1, 1, 1, 0},
        {0, 1,-1, 2,-2, 0},
        {0, 1, 1, 4, 4, 0},
        {0, 1,-1, 8,-8, 1}};
    float tmp[4][6];
    // tmp = AT * M
    for (int c = 0; c < 6; c++) {
        float m0=M[0*6+c], m1=M[1*6+c], m2=M[2*6+c], m3=M[3*6+c], m4=M[4*6+c], m5=M[5*6+c];
        for (int i = 0; i < 4; i++)
            tmp[i][c] = AT[i][0]*m0 + AT[i][1]*m1 + AT[i][2]*m2 + AT[i][3]*m3 + AT[i][4]*m4 + AT[i][5]*m5;
    }
    // Y = tmp * AT^T
    for (int i = 0; i < 4; i++) {
        float t0=tmp[i][0], t1=tmp[i][1], t2=tmp[i][2], t3=tmp[i][3], t4=tmp[i][4], t5=tmp[i][5];
        for (int j = 0; j < 4; j++)
            Y[i*4+j] = AT[j][0]*t0 + AT[j][1]*t1 + AT[j][2]*t2 + AT[j][3]*t3 + AT[j][4]*t4 + AT[j][5]*t5;
    }
}

// im2col: �� NCHW ����ת�����о���
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

// im2col��K ��㲼�֣���output Ϊ [C*kH*kW, N*outH*outW]���� [K, M]
// �� gemm_nm �� X ���벼��һ�£�GEMM �׶�����ת��
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

// Fast depthwise 3x3 s1 p1 convolution (group == C_in == C_out, 1 ch/group).
// C2PSA attention depthwise convs previously fell into the generic group
// path: sequential over ALL channels with an im2col+gemm per channel
// (~18ms/frame on v11m256). Here: pad per channel + AVX2 across 8 columns,
// channels parallelized via the thread pool.
static inline void depthwise3x3s1(const float* input, const float* weight, const float* bias,
                                  float* output, bool silu, int C, int H, int W) {
    const int rows = H * W;
    const int pH = H + 2, pW = W + 2;
    float* pad = get_temp_buf((size_t)C * pH * pW + 8);
    SimpleThreadPool& pool = SimpleThreadPool::instance();
    int workers = pool.num_threads();
    bool mt = workers > 1 && (size_t)C * rows > (size_t)4096;

    auto body = [&](int c0, int c1) {
        for (int c = c0; c < c1; c++) {
            const float* in_c = input + (size_t)c * rows;
            float* pad_c = pad + (size_t)c * pH * pW;
            const float* wp = weight + (size_t)c * 9;
            float* out_c = output + (size_t)c * rows;
            memset(pad_c, 0, sizeof(float) * pH * pW);
            for (int y = 0; y < H; y++)
                memcpy(pad_c + (size_t)(y + 1) * pW + 1, in_c + (size_t)y * W, sizeof(float) * W);
            const float b = bias ? bias[c] : 0.0f;
            const float w00 = wp[0], w01 = wp[1], w02 = wp[2];
            const float w10 = wp[3], w11 = wp[4], w12 = wp[5];
            const float w20 = wp[6], w21 = wp[7], w22 = wp[8];
            const __m256 vb = _mm256_set1_ps(b);
            const __m256 vw00 = _mm256_set1_ps(w00), vw01 = _mm256_set1_ps(w01), vw02 = _mm256_set1_ps(w02);
            const __m256 vw10 = _mm256_set1_ps(w10), vw11 = _mm256_set1_ps(w11), vw12 = _mm256_set1_ps(w12);
            const __m256 vw20 = _mm256_set1_ps(w20), vw21 = _mm256_set1_ps(w21), vw22 = _mm256_set1_ps(w22);
            for (int y = 0; y < H; y++) {
                const float* r0 = pad_c + (size_t)y * pW;
                const float* r1 = r0 + pW;
                const float* r2 = r1 + pW;
                float* o = out_c + (size_t)y * W;
                int x = 0;
                for (; x + 8 <= W; x += 8) {
                    __m256 acc = vb;
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x), vw00, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 1), vw01, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + x + 2), vw02, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x), vw10, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 1), vw11, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + x + 2), vw12, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x), vw20, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 1), vw21, acc);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + x + 2), vw22, acc);
                    if (silu) {
                        __m256 t = _mm256_sub_ps(_mm256_setzero_ps(), acc);
                        t = _mm256_min_ps(t, _mm256_set1_ps(86.0f));
                        t = _mm256_max_ps(t, _mm256_set1_ps(-86.0f));
                        t = _mm256_fmadd_ps(t, _mm256_set1_ps(12102203.0f), _mm256_set1_ps(1064866805.0f));
                        __m256 ex = _mm256_castsi256_ps(_mm256_cvttps_epi32(t));
                        acc = _mm256_div_ps(acc, _mm256_add_ps(_mm256_set1_ps(1.0f), ex));
                    }
                    _mm256_storeu_ps(o + x, acc);
                }
                for (; x < W; x++) {
                    float v = b + r0[x] * w00 + r0[x + 1] * w01 + r0[x + 2] * w02
                                 + r1[x] * w10 + r1[x + 1] * w11 + r1[x + 2] * w12
                                 + r2[x] * w20 + r2[x + 1] * w21 + r2[x + 2] * w22;
                    o[x] = silu ? v / (1.0f + fast_exp(-v)) : v;
                }
            }
        }
    };

    if (mt) {
        int qc = std::max(1, C / (workers * 2));
        int qb = (C + qc - 1) / qc;
        pool.parallel_for(qb, [&](int qi) { body(qi * qc, std::min(C, (qi + 1) * qc)); });
    } else {
        body(0, C);
    }
}

static inline void op_conv(const Tensor& input, const Tensor& weight, const Tensor* bias,
                           Tensor& output, const ConvAttr& attr, bool apply_silu = false) {
    // 调试开关：TINY_YOLO_NO_SILU=1 时禁用所有融合 SiLU（定位数值问题用）
    static const bool g_no_silu = (getenv("TINY_YOLO_NO_SILU") != nullptr);
    if (g_no_silu) apply_silu = false;
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

    // NaN 定位探针：打印首�?Conv 的实际属性，确认分发路径
    {
        static const bool nan_probe0 = (getenv("TINY_YOLO_NAN_DEBUG") != nullptr);
        static int probe_count = 0;
        if (nan_probe0 && probe_count < 2) {
            probe_count++;
            printf("[conv] k=%dx%d s=%d,%d p=%d,%d dil=%d,%d group=%d | in=%dx%dx%dx%d w=%d -> out=%dx%d\n",
                   kH, kW, attr.stride_h, attr.stride_w, attr.pad_h, attr.pad_w,
                   attr.dilation_h, attr.dilation_w, group, N, C_in, H, W, C_out, outH, outW);
        }
    }

    if (group == 1) {
        int col_size = C_in * kH * kW;
        int M = outH * outW;

        // 3x3 s1 p1��Winograd��SiLU �ں�������任�׶Σ�?
        bool is_3x3s1p1 = (kH == 3 && kW == 3 && attr.stride_h == 1 && attr.stride_w == 1
                            && attr.pad_h == 1 && attr.pad_w == 1 && attr.dilation_h == 1 && attr.dilation_w == 1);

        if (is_3x3s1p1) {
            conv3x3s1_winograd(input.data, weight.data, bias ? bias->data : nullptr,
                        output.data, apply_silu, N, C_in, H, W, C_out);
            return;
        }

        // 1x1 s1 p0�����뱾������ [K=C_in, M] ���֣�ֱ�� GEMM������ת��
        bool is_1x1 = (kH == 1 && kW == 1 && attr.stride_h == 1 && attr.stride_w == 1
                        && attr.pad_h == 0 && attr.pad_w == 0 && attr.dilation_h == 1 && attr.dilation_w == 1);

        if (is_1x1) {
            for (int n = 0; n < N; n++) {
                const float* in_n = input.data + (size_t)n * C_in * M;
                float* out_n = output.data + (size_t)n * C_out * M;
                gemm_nm(in_n, weight.data, bias ? bias->data : nullptr,
                        out_n, C_out, M, C_in, apply_silu);
            }
        // ע��3x3 s2 ֱ�Ӿ���·�����޸���������ԭʵ������������SiLU �� q ѭ������ͨ��
        // Ӧ�ã�AVX �ڲ㰴 s1 ��ʽ�������أ�δ���� stride=2 ʱ����������������?2����
        // ʵ���� Ryzen 5560U �϶������Բ��� im2col_km + gemm_nm(AVX2) ��ϣ���Ĭ�Ͻ��ã�?
        // �������ã��ָ��·� else-if ������ȥ�� false &&������ test_s2.exe ��֤��
        } else {
            // 普通卷积：im2col �?[K, M] 布局 + gemm_nm
            size_t col_numel = (size_t)N * M * col_size;
            float* col = get_temp_buf(col_numel);
            im2col_km(input.data, col, N, C_in, H, W, kH, kW,
                      attr.stride_h, attr.stride_w, attr.pad_h, attr.pad_w,
                      attr.dilation_h, attr.dilation_w, outH, outW);

            // NaN 定位探针（TINY_YOLO_NAN_DEBUG=1）：检�?im2col 输出与各阶段结果
            static const bool nan_probe = (getenv("TINY_YOLO_NAN_DEBUG") != nullptr);
            if (nan_probe) {
                size_t cn = 0;
                for (size_t q = 0; q < col_numel; q++) if (_isnan(col[q])) cn++;
                if (cn) printf("[probe] after im2col: col nan=%zu (col_numel=%zu)\n", cn, col_numel);
                // 检查输�?
                size_t inan = 0;
                size_t in_numel = (size_t)N * C_in * H * W;
                for (size_t q = 0; q < in_numel; q++) if (_isnan(input.data[q])) inan++;
                if (inan) printf("[probe] conv input nan=%zu\n", inan);
            }

            for (int n = 0; n < N; n++) {
                float* col_n = col + (size_t)n * M * col_size;
                float* out_n = output.data + (size_t)n * C_out * M;
                gemm_nm(col_n, weight.data, bias ? bias->data : nullptr,
                        out_n, C_out, M, col_size, apply_silu);
                if (nan_probe) {
                    size_t on = 0; int bad_ch = -1, nbad_ch = 0;
                    for (int c = 0; c < C_out; c++) {
                        size_t cn2 = 0;
                        for (int q = 0; q < M; q++) if (_isnan(out_n[(size_t)c * M + q])) cn2++;
                        if (cn2) { nbad_ch++; if (bad_ch < 0) bad_ch = c; on += cn2; }
                    }
                    if (on) printf("[probe] after gemm+silu: out nan=%zu in %d channels (first ch=%d)\n", on, nbad_ch, bad_ch);
                }
            }
        }
    } else {
        // ������������鴦����?
        int C_per_group = C_in / group;
        int C_out_per_group = C_out / group;
        int col_size = C_per_group * kH * kW;

        // depthwise fast path: 1 channel per group, 3x3 stride-1
        // (typical C2PSA attention depthwise convs, e.g. v11m/v26)
        if (C_per_group == 1 && C_out_per_group == 1 && group == C_in && group == C_out
            && kH == 3 && kW == 3 && attr.stride_h == 1 && attr.stride_w == 1
            && attr.pad_h == 1 && attr.pad_w == 1
            && attr.dilation_h == 1 && attr.dilation_w == 1) {
            depthwise3x3s1(input.data, weight.data, bias ? bias->data : nullptr,
                           output.data, apply_silu, C_in, H, W);
            return;
        }
        int M = outH * outW;
        size_t col_numel = (size_t)N * M * col_size;
        size_t group_input_numel = (size_t)N * C_per_group * H * W;
        size_t max_numel = col_numel > group_input_numel ? col_numel : group_input_numel;

        float* col = get_temp_buf(max_numel);
        float* group_input = col + col_numel;  // ���û������ĺ�벿�֣��������
        bool need_separate = (col_numel + group_input_numel > max_numel);
        if (need_separate) {
            // �����󣬵�������
            group_input = (float*)malloc(group_input_numel * sizeof(float));
        }

        for (int g = 0; g < group; g++) {
            // ��ȡ���������?
            for (int n = 0; n < N; n++)
                for (int c = 0; c < C_per_group; c++)
                    memcpy(group_input + (n * C_per_group + c) * H * W,
                           input.data + (n * C_in + g * C_per_group + c) * H * W,
                           H * W * sizeof(float));

            im2col(group_input, col, N, C_per_group, H, W, kH, kW,
                   attr.stride_h, attr.stride_w, attr.pad_h, attr.pad_w,
                   attr.dilation_h, attr.dilation_w, outH, outW);

            // �����Ȩ��?
            const float* group_weight = weight.data + (size_t)g * C_out_per_group * col_size;
            const float* group_bias = bias ? bias->data + g * C_out_per_group : nullptr;

            for (int n = 0; n < N; n++) {
                float* col_n = col + (size_t)n * M * col_size;
                float* out_n = output.data + (size_t)n * C_out * M + (size_t)g * C_out_per_group * M;
                // ʹ�� gemm_nc ֱ�����?[C_out_per_group, M] ����
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
// �����
// ============================================================
// ���� exp ���ƣ�Schraudolph�����������㹻�����ã���expf��ܶ�?
static inline float fast_exp(float x) {
    // 注意：Schraudolph 近似�?x < -87.9895 时整数结果变负，reinterpret �?float 会得�?
    // 指数位全 1 �?NaN（毒区）。钳制阈值必须留足余量（86 距毒区约 2.0），否则
    // 大激活值经 SiLU/sigmoid 会产�?NaN 并向下游扩散（yolo12n 自定义模型曾踩中�?
    if (x > 86.0f) return 1e30f;
    if (x < -86.0f) return 0.0f;
    // Schraudolph���ƣ�����IEEE754��������ָ��λ
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
// Ԫ�ؼ����㣨֧�ֹ㲥��
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
    // ������ı�ƽ����ת��Ϊ����ı�ƽ�����������㲥��
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

    // ����·�� 1����״��ͬ��ֱ����Ԫ������
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

    // ����·�� 2��a �Ǳ���
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

    // ����·�� 3��b �Ǳ���
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

    // ͨ��·�����㲥
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
    // �淶�� axis��֧�ָ�����
    int ndim = inputs[0]->ndim;
    if (axis < 0) axis += ndim;

    std::vector<int> out_shape = inputs[0]->shape;
    int total_axis = 0;
    for (auto* inp : inputs) total_axis += inp->shape[axis];
    out_shape[axis] = total_axis;
    output.alloc(out_shape);

    // ����ÿ������������е�ƫ��?
    int offset = 0;
    SimpleThreadPool& pool_ = SimpleThreadPool::instance();
    bool cmt = (pool_.num_threads() > 1 && (size_t)output.numel * sizeof(float) >= 262144);
    for (auto* inp : inputs) {
        int in_axis_size = inp->shape[axis];
        // ��Ԫ�ظ���
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
// Resize (˫���Բ�ֵ�������Ҳ֧��?
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

    // �ػ�·����2 ��������ϲ�����YOLO �ϲ�������������?
    // ÿ��Դ��ˮƽ���� 2 ��������д���飬�ڴ��������?
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
                        // �����?
                        int ih = std::min((int)(oh * scaleH), H - 1);
                        int iw = std::min((int)(ow * scaleW), W - 1);
                        output.at(n, c, oh, ow) = input.at(n, c, ih, iw);
                    } else {
                        // ˫���� (ONNX half_pixel / PyTorch align_corners=False)
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
// AVX2 8x8 float ��ת�ã�in[r][c] -> out[c][r]
// in �������о� lda����out �������о� ldb��
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

    // ����·����3D ����ת�� perm=[0,2,1]��ע�����鳣����
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
            // ��β���� 8 ������������ c ѭ�����ǣ����ﲹ�п�β��
        };
        int rb = (R + 7) / 8;
        // �з���Ҳ�ֿ����������ж�
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

    // ����·����4D �����ĳ���ת��ģʽ
    if (input.ndim == 4 && perm.size() == 4) {
        int N = input.shape[0], C = input.shape[1], H = input.shape[2], W = input.shape[3];
        const float* in = input.data;
        float* out = output.data;
        SimpleThreadPool& pool_ = SimpleThreadPool::instance();
        bool tmt = (pool_.num_threads() > 1 && (size_t)input.numel >= 65536);

        // perm = [0,2,3,1]: NCHW -> NHWC���� (n,h) ����п鲢�У�?
        if (perm[0]==0 && perm[1]==2 && perm[2]==3 && perm[3]==1) {
            int rows = N * H;
            auto body = [&](int r) {
                int n = r / H, h = r % H;
                const float* in_row = in + (size_t)n * C * H * W + (size_t)h * W;
                float* out_row = out + ((size_t)n * H + h) * W * C;
                // 8x8 ��ת�ã�8 ͨ�� �� 8 �� w
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
        // perm = [0,3,1,2]: NHWC -> NCHW��N=1 ʱ��ͨ����Χ�ֿ飩
        if (perm[0]==0 && perm[1]==3 && perm[2]==1 && perm[3]==2) {
            int H2 = input.shape[1], W2 = input.shape[2], C2 = input.shape[3];
            size_t plane = (size_t)H2 * W2;
            // 8x8 ��ת�ã�8 �� i(=h*W+w) �� 8 ͨ��
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
        // perm = [0,1,3,2]: ת�������ά����ͨ���в��У�?
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

    // ͨ��·�����ò������㣨�ȱ�ƽ����+ȡģ�죩
    std::vector<int> in_strides(input.ndim);
    int s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = s; s *= input.shape[i]; }

    std::vector<int> out_strides(output.ndim);
    s = 1;
    for (int i = output.ndim - 1; i >= 0; i--) { out_strides[i] = s; s *= output.shape[i]; }

    // ���� perm ����ӳ�䣺in_dim_to_out_dim
    std::vector<int> inv_perm(input.ndim);
    for (size_t i = 0; i < perm.size(); i++) inv_perm[perm[i]] = (int)i;

    // ��Ƕ��ѭ��ת�ã��ݹ�չ����
    // �򻯣��ö�ά����������
    std::vector<int> out_idx(output.ndim, 0);
    for (int i = 0; i < output.numel; i++) {
        // ���������ƽ����?
        int in_flat = 0;
        for (int d = 0; d < input.ndim; d++) {
            in_flat += out_idx[inv_perm[d]] * in_strides[d];
        }
        output.data[i] = input.data[in_flat];

        // ���������ά����?
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
    // ������Ԫ���������� -1 �� 0��ONNX��0��ʾ����ԭά�ȣ�
    std::vector<int> shape = new_shape;

    // �ȴ���0ά���������и��ƶ�Ӧά��
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

    output.reference(shape, input.data); // ����ͬһ���ڴ�
    output.own_data = false;
}

// MatMul ���а�װ���� M �зֿ鲢�У����С����?matmul_avx �� 4 ��չ����
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
    // ֧�� 2D �͸�ά batch matmul
    int a_ndim = a.ndim, b_ndim = b.ndim;
    if (a_ndim == 2 && b_ndim == 2) {
        int M = a.shape[0], K = a.shape[1], N = b.shape[1];
        output.alloc({M, N});
        matmul_par(a.data, b.data, nullptr, output.data, M, N, K);
    } else {
        // ��ά����ǰ���ά�ȵ���?batch
        int K = a.shape[a_ndim - 1];
        int M = a.shape[a_ndim - 2];
        int N = b.shape[b_ndim - 1];

        // ���� batch �����㲥��
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

        // ��������ʱ�� batch ����
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
            // �ҵ� a �� b ��Ӧ�� batch �����������㲥��
            // a ������ batch ά�ȣ�b ������ 2D������ batch ������
            int a_batch_idx = b_idx;
            int b_batch_idx = b_idx;
            
            // ���� a ���� batch ��С
            int a_total_batch = 1;
            for (int i = 0; i < a_ndim - 2; i++) a_total_batch *= a.shape[i];
            // ���� b ���� batch ��С
            int b_total_batch = 1;
            for (int i = 0; i < b_ndim - 2; i++) b_total_batch *= b.shape[i];
            
            // ���?b û�� batch ά�ȣ�2D�������� batch ����ͬһ�� b
            if (b_total_batch == 1) b_batch_idx = 0;
            // ���?a û�� batch ά�ȣ����� batch ����ͬһ�� a
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

    // ����ÿ�������?axis ά�ȵĴ�С�����ȼ��������split_sizes > ����������е����?> �ȷ֣�
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
            // �ȷ�
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
    // ��һ�� starts/ends�������������� clamp
    std::vector<int> norm_starts = starts, norm_ends = ends;
    for (size_t i = 0; i < axes.size(); i++) {
        int ax = axes[i] < 0 ? axes[i] + input.ndim : axes[i];
        int dim = input.shape[ax];
        int step = i < steps.size() ? steps[i] : 1;
        // ����������
        if (norm_starts[i] < 0) norm_starts[i] += dim;
        if (norm_ends[i] < 0) norm_ends[i] += dim;
        // clamp
        norm_starts[i] = std::max(0, std::min(norm_starts[i], dim));
        norm_ends[i] = std::max(0, std::min(norm_ends[i], dim));
        out_shape[ax] = (norm_ends[i] - norm_starts[i] + step - 1) / step;
    }
    output.alloc(out_shape);

    // ��ʵ�֣���Ԫ�ظ���
    for (int i = 0; i < output.numel; i++) {
        // �����ƽ����?-> �����ά����?
        int idx = i;
        std::vector<int> out_idx(output.ndim);
        for (int d = output.ndim - 1; d >= 0; d--) {
            out_idx[d] = idx % output.shape[d];
            idx /= output.shape[d];
        }
        // ת��Ϊ��������
        std::vector<int> in_idx = out_idx;
        for (size_t j = 0; j < axes.size(); j++) {
            int ax = axes[j] < 0 ? axes[j] + input.ndim : axes[j];
            int step = j < steps.size() ? steps[j] : 1;
            in_idx[ax] = norm_starts[j] + out_idx[ax] * step;
        }
        // ���������ƽ����?
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

    // �Ż�ʵ�֣��ò�����ѭ��Ƕ��
    // �������벽��
    std::vector<int> in_strides(input.ndim);
    int s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = s; s *= input.shape[i]; }

    // �����������?
    std::vector<int> out_strides(output.ndim);
    s = 1;
    for (int i = output.ndim - 1; i >= 0; i--) { out_strides[i] = s; s *= output.shape[i]; }

    // ���� reduce ά�ȵ��ܴ�С�Ͳ���
    int reduce_size = 1;
    for (int d = 0; d < input.ndim; d++)
        if (reduce_dim[d]) reduce_size *= input.shape[d];

    // Ԥ����ÿ�� reduce ������Ӧ������ƫ��
    // �򻯣��õݹ�����ö�� reduce ά�����?
    // ���ڳ��������reduce ���һά������ά�ȣ������Ը����?

    // ͨ��ʵ�֣��ö�ά����������
    std::vector<int> out_idx(output.ndim, 0);
    std::vector<int> in_idx(input.ndim, 0);

    for (int i = 0; i < output.numel; i++) {
        // ���������������� reduce ά�ȴ��������ӳ��?
        int out_d = 0;
        for (int d = 0; d < input.ndim; d++) {
            if (!reduce_dim[d]) {
                in_idx[d] = out_idx[out_d++];
            }
        }

        // �����������ƫ�ƣ���?reduce ά�ȣ�
        int base_offset = 0;
        for (int d = 0; d < input.ndim; d++) {
            if (!reduce_dim[d]) base_offset += in_idx[d] * in_strides[d];
        }

        // ö�� reduce ά�ȵ��������?
        float result = is_max ? -1e30f : 0.0f;

        // �õݹ�����ö�� reduce ά��
        // �򻯣��ñ�ƽ���� + ���루��ֻ�� reduce ά�ȣ�
        std::vector<int> red_axes;
        for (int d = 0; d < input.ndim; d++)
            if (reduce_dim[d]) red_axes.push_back(d);

        std::vector<int> red_idx(red_axes.size(), 0);
        for (int r = 0; r < reduce_size; r++) {
            // ��������ƫ��
            int offset = base_offset;
            for (size_t j = 0; j < red_axes.size(); j++) {
                offset += red_idx[j] * in_strides[red_axes[j]];
            }

            float v = input.data[offset];
            if (is_max) result = std::max(result, v);
            else result += v;

            // ���� reduce ����
            for (int j = (int)red_axes.size() - 1; j >= 0; j--) {
                red_idx[j]++;
                if (red_idx[j] < input.shape[red_axes[j]]) break;
                red_idx[j] = 0;
            }
        }

        output.data[i] = result;

        // �����������?
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

    // ����·����inner==1��reduce���һά���������ڴ����
    if (inner == 1) {
        const bool vok = axis_size >= 16;
        const __m256 vones = _mm256_set1_ps(1.0f);
        for (int o = 0; o < outer; o++) {
            const float* src = input.data + (size_t)o * axis_size;
            float* dst = output.data + (size_t)o * axis_size;
            if (vok) {
                // AVX2��max / ������ Schraudolph exp / sum / ��һ��
                __m256 vmax = _mm256_set1_ps(-1e30f);
                int a = 0;
                for (; a + 8 <= axis_size; a += 8)
                    vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(src + a));
                float max_val = hmax256_ps(vmax);
                for (; a < axis_size; a++) max_val = std::max(max_val, src[a]);
                __m256 vsum = _mm256_setzero_ps();
                const __m256 vlo86 = _mm256_set1_ps(-86.0f), vhi86 = _mm256_set1_ps(86.0f);
                const __m256 vscale = _mm256_set1_ps(12102203.0f), vbias = _mm256_set1_ps(1064866805.0f);
                for (a = 0; a + 8 <= axis_size; a += 8) {
                    __m256 t = _mm256_sub_ps(_mm256_loadu_ps(src + a), _mm256_set1_ps(max_val));
                    t = _mm256_min_ps(_mm256_max_ps(t, vlo86), vhi86);
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
                // ������?
                float max_val = src[0];
                for (int a = 1; a < axis_size; a++) {
                    if (src[a] > max_val) max_val = src[a];
                }
                // ���� exp �� sum
                float sum = 0;
                for (int a = 0; a < axis_size; a++) {
                    float v = fast_exp(src[a] - max_val);
                    dst[a] = v;
                    sum += v;
                }
                // ��һ��
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
            // ������?
            float max_val = -1e30f;
            for (int a = 0; a < axis_size; a++) {
                int idx = ((o * axis_size + a) * inner) + i;
                max_val = std::max(max_val, input.data[idx]);
            }
            // ���� exp �� sum
            float sum = 0;
            for (int a = 0; a < axis_size; a++) {
                int idx = ((o * axis_size + a) * inner) + i;
                float v = fast_exp(input.data[idx] - max_val);
                output.data[idx] = v;
                sum += v;
            }
            // ��һ��
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
    // ����������?
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

    // ����������������
    std::vector<int> in_strides(input.ndim), out_strides(input.ndim);
    int in_s = 1, out_s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) {
        in_strides[i] = in_s; in_s *= input.shape[i];
        out_strides[i] = out_s; out_s *= out_shape[i];
    }

    // ��Ԫ�ظ���
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

    // ����������?
    std::vector<int> out_shape;
    for (int i = 0; i < axis; i++) out_shape.push_back(input.shape[i]);
    for (int i = 0; i < indices.ndim; i++) out_shape.push_back(indices.shape[i]);
    for (int i = axis + 1; i < input.ndim; i++) out_shape.push_back(input.shape[i]);
    output.alloc(out_shape);

    // ���㲽��
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

    // ���㲽��
    std::vector<int> in_strides(input.ndim), out_strides(indices.ndim);
    int in_s = 1, out_s = 1;
    for (int i = input.ndim - 1; i >= 0; i--) { in_strides[i] = in_s; in_s *= input.shape[i]; }
    for (int i = indices.ndim - 1; i >= 0; i--) { out_strides[i] = out_s; out_s *= indices.shape[i]; }

    for (int i = 0; i < output.numel; i++) {
        // �����������?
        std::vector<int> out_idx(indices.ndim);
        int rem = i;
        for (int d = 0; d < indices.ndim; d++) {
            out_idx[d] = rem / out_strides[d];
            rem %= out_strides[d];
        }
        // ��������ƫ�ƣ�axisά����indices��ֵ��
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

    // ������?
    std::vector<int> out_shape = input.shape;
    out_shape[axis] = k;
    output_values.alloc(out_shape);
    output_indices.alloc(out_shape);

    // ���㲽��
    int outer = 1, inner = 1;
    for (int i = 0; i < axis; i++) outer *= input.shape[i];
    for (int i = axis + 1; i < input.ndim; i++) inner *= input.shape[i];

    for (int o = 0; o < outer; o++) {
        for (int inn = 0; inn < inner; inn++) {
            // �ռ�axisά�ȵ�����ֵ
            std::vector<std::pair<float, int>> vec(axis_size);
            for (int a = 0; a < axis_size; a++) {
                int offset = (o * axis_size + a) * inner + inn;
                vec[a] = {input.data[offset], a};
            }
            // ����
            if (largest) {
                std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
            } else {
                std::sort(vec.begin(), vec.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
            }
            // ȡǰk��
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

    // ͨ�ù㲥
    for (int i = 0; i < output.numel; i++) {
        float va = read_tensor_val(a, broadcast_index(i, out_shape, a.shape));
        float vb = read_tensor_val(b, broadcast_index(i, out_shape, b.shape));
        output.data[i] = va - vb * floorf(va / vb);
    }
}

// ============================================================
// Cast (Ŀǰֻ֧��float��ֱ�Ӹ���)
// ============================================================
static inline void op_cast(const Tensor& input, Tensor& output) {
    output.alloc(input.shape);
    memcpy(output.data, input.data, input.numel * sizeof(float));
}

