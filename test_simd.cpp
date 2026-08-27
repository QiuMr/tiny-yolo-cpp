// Standalone unit test: SIMD batched F(4,3) transforms vs scalar (from ops.h)
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>

// pull the shared header only for the scalar transform implementations
#include "tiny_yolo/ops.h"

static unsigned g_seed = 12345u;
static float rndf() {
    g_seed = g_seed * 1664525u + 1013904223u;
    return ((g_seed >> 8) & 0xFFFF) / 65536.0f * 2.0f - 1.0f; // [-1,1]
}

int main() {
    int failures = 0;
    double max_diff_in = 0, max_diff_out = 0;
    const int TS = 4, KT = 36;
    const int w_tiles = 8, h_tiles = 8;
    const int T = w_tiles * h_tiles;         // 64
    const int pW = w_tiles * TS + 2, pH = h_tiles * TS + 2; // padded size

    // ---- input transform ----
    std::vector<float> padded(pH * pW);
    for (auto& v : padded) v = rndf();
    std::vector<float> ref(T * KT), got(T * KT);

    for (int t = 0; t < T; t++) { // scalar reference
        int ht = t / w_tiles, wt = t % w_tiles;
        int oh = ht * TS, ow = wt * TS;
        const float* d_src = padded.data() + oh * pW + ow;
        float d_block[36], v[36];
        for (int r = 0; r < 6; r++)
            memcpy(d_block + r * 6, d_src + r * pW, 6 * sizeof(float));
        wino43_input_transform(d_block, v);
        for (int kk = 0; kk < KT; kk++) ref[(size_t)t * KT + kk] = v[kk];
    }
    for (int t0 = 0; t0 + 8 <= T; t0 += 8)
        wino43_input_transform8(padded.data(), got.data(), T, w_tiles, pW, t0);
    for (size_t i = 0; i < ref.size(); i++) {
        double d = std::fabs((double)ref[i] - got[i]);
        if (d > max_diff_in) max_diff_in = d;
        if (d > 1e-3) { failures++; if (failures < 5) printf("IN MISMATCH idx=%zu ref=%.6f got=%.6f\n", i, ref[i], got[i]); }
    }

    // ---- output transform ----
    const int C_out = 3;
    std::vector<float> Mbuf((size_t)KT * C_out * T);
    for (auto& v : Mbuf) v = rndf();
    std::vector<float> out_ref((size_t)C_out * TS * TS * T), out_got(out_ref.size());
    const float bias[3] = {0.3f, -0.2f, 0.0f};

    // scalar per (p,t) with silu=false
    for (int p = 0; p < C_out; p++) {
        for (int t = 0; t < T; t++) {
            int ht = t / w_tiles, wt = t % w_tiles;
            int oh = ht * TS, ow = wt * TS;
            float m[36], y[16];
            for (int k = 0; k < KT; k++) m[k] = Mbuf[((size_t)k * C_out + p) * T + t];
            wino43_output_transform(m, y);
            for (int i = 0; i < 16; i++) y[i] += bias[p];
            for (int r = 0; r < TS && oh + r < TS * h_tiles; r++)
                for (int s = 0; s < TS && ow + s < TS * w_tiles; s++)
                    out_ref[(size_t)p * ((size_t)TS * h_tiles * TS * w_tiles) + (size_t)(oh + r) * (TS * w_tiles) + (ow + s)] = y[r * TS + s];
        }
    }
    for (int p = 0; p < C_out; p++)
        for (int t0 = 0; t0 + 8 <= T; t0 += 8)
            wino43_output_transform8(Mbuf.data(), out_got.data(), bias, false,
                                     C_out, p, T, w_tiles, TS * h_tiles, TS * w_tiles, t0);
    for (size_t i = 0; i < out_ref.size(); i++) {
        double d = std::fabs((double)out_ref[i] - out_got[i]);
        if (d > max_diff_out) max_diff_out = d;
        if (d > 1e-3) { failures++; if (failures < 10) printf("OUT MISMATCH idx=%zu ref=%.6f got=%.6f\n", i, out_ref[i], out_got[i]); }
    }

    printf("input  maxdiff = %.3e  %s\n", max_diff_in, max_diff_in < 1e-3 ? "OK" : "FAIL");
    printf("output maxdiff = %.3e  %s\n", max_diff_out, max_diff_out < 1e-3 ? "OK" : "FAIL");
    printf("total failures: %d\n", failures);
    return failures ? 1 : 0;
}