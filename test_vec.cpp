// Correctness checks for the AVX2-vectorized operator paths: compare each
// vectorized op against a plain scalar reference on data that actually
// exercises the vector loop (wide enough that the SIMD path runs).
#define _CRT_SECURE_NO_WARNINGS
// include ops.h first: it sets NOMINMAX before pulling in windows.h
#include "tiny_yolo/ops.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>

static int g_fail = 0;
static void chk(const char* name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (!ok) g_fail++;
}

static void ref_maxpool(const float* in, float* out, int C, int H, int W,
                        int kH, int kW, int sH, int sW, int pH, int pW,
                        int outH, int outW) {
    for (int c = 0; c < C; c++) {
        const float* ic = in + (size_t)c * H * W;
        float* oc = out + (size_t)c * outH * outW;
        for (int oh = 0; oh < outH; oh++)
            for (int ow = 0; ow < outW; ow++) {
                float m = -1e30f;
                for (int kh = 0; kh < kH; kh++)
                    for (int kw = 0; kw < kW; kw++) {
                        int ih = oh * sH - pH + kh, iw = ow * sW - pW + kw;
                        if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                            m = std::max(m, ic[ih * W + iw]);
                    }
                oc[oh * outW + ow] = m;
            }
    }
}

static bool maxpool_case(const char* label, int H, int W, int kH, int kW,
                         int sH, int sW, int pH, int pW) {
    const int C = 2;
    std::vector<float> d((size_t)C * H * W);
    for (size_t i = 0; i < d.size(); i++) d[i] = (float)((int)(i * 37 % 101) - 50);

    int outH = (H + 2 * pH - kH) / sH + 1;
    int outW = (W + 2 * pW - kW) / sW + 1;

    Tensor in, out;
    in.reference({1, C, H, W}, d.data());
    op_maxpool(in, out, kH, kW, sH, sW, pH, pW);

    std::vector<float> ref((size_t)C * outH * outW);
    ref_maxpool(d.data(), ref.data(), C, H, W, kH, kW, sH, sW, pH, pW, outH, outW);

    int bad = 0;
    for (size_t i = 0; i < ref.size(); i++) if (out.data[i] != ref[i]) bad++;
    printf("      %-14s out=%dx%d  mismatches=%d/%d\n", label, outH, outW, bad, (int)ref.size());
    return bad == 0;
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("op_maxpool\n");
    chk("5x5 s1 p2 (SPPF, what all target models use)",
        maxpool_case("5x5 s1 p2", 16, 32, 5, 5, 1, 1, 2, 2));
    chk("2x2 s2 p0 (stride > 1)",
        maxpool_case("2x2 s2 p0", 8, 32, 2, 2, 2, 2, 0, 0));
    chk("3x3 s2 p1 (stride > 1)",
        maxpool_case("3x3 s2 p1", 16, 32, 3, 3, 2, 2, 1, 1));

    printf("activations (vectorized vs scalar fast_exp)\n");
    {
        const int n = 1000;
        std::vector<float> d(n);
        for (int i = 0; i < n; i++) d[i] = (float)(i - 500) * 0.05f;
        Tensor in; in.reference({1, 1, 1, n}, d.data());
        Tensor o1, o2, o3, o4;
        op_silu(in, o1); op_sigmoid(in, o2); op_relu(in, o3); op_exp(in, o4);
        int bs = 0, bg = 0, br = 0, be = 0;
        // The vectorized sites evaluate 12102203*x + 1064866805 with one FMA
        // while scalar fast_exp does mul+add (two roundings). Around x~25 the
        // float intermediate is ~1.4e9, where one ULP is already ~128, so the
        // two forms can land on int results a few hundred apart -> up to ~1e-4
        // of relative difference. That split is inherent to the approximation
        // and predates this branch (6 AVX2 exp sites, gemm fused-SiLU included,
        // already used FMA); it has no measurable effect on results -- detection
        // boxes are bit-identical. 1e-4 is still 300x tighter than the ~3%
        // accuracy of Schraudolph's approximation itself.
        // (named approx_eq, not "near": windows.h still defines near/far macros)
        auto approx_eq = [](float x, float y) {
            float m = std::max(1.0f, std::max(std::fabs(x), std::fabs(y)));
            return std::fabs(x - y) <= 1e-4f * m;
        };
        for (int i = 0; i < n; i++) {
            float x = d[i];
            if (!approx_eq(o1.data[i], x / (1.0f + fast_exp(-x)))) bs++;
            if (!approx_eq(o2.data[i], 1.0f / (1.0f + fast_exp(-x)))) bg++;
            if (o3.data[i] != (x > 0 ? x : 0.0f)) br++;  // relu is exact
            if (!approx_eq(o4.data[i], fast_exp(x))) be++;
        }
        printf("      out-of-tolerance silu=%d sigmoid=%d relu=%d exp=%d / %d\n", bs, bg, br, be, n);
        chk("op_silu matches scalar within 1e-4 rel", bs == 0);
        chk("op_sigmoid matches scalar within 1e-4 rel", bg == 0);
        chk("op_relu matches scalar exactly", br == 0);
        chk("op_exp matches scalar within 1e-4 rel", be == 0);
    }

    printf("op_elementwise broadcast (odometer walk)\n");
    {
        const int N = 2, C = 3, H = 4, W = 5;
        std::vector<float> a((size_t)N * C * H * W), b((size_t)N * C);
        for (size_t i = 0; i < a.size(); i++) a[i] = (float)(i % 17) - 8.0f;
        for (size_t i = 0; i < b.size(); i++) b[i] = (float)(i + 1) * 0.5f;

        Tensor ta, tb, out;
        ta.reference({N, C, H, W}, a.data());
        tb.reference({N, C, 1, 1}, b.data());
        op_elementwise(ta, tb, out, '*');

        int bad = 0;
        for (int n = 0; n < N; n++)
            for (int c = 0; c < C; c++)
                for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++) {
                        float ref = a[((size_t)n * C + c) * H * W + h * W + w] * b[(size_t)n * C + c];
                        if (out.data[((size_t)n * C + c) * H * W + h * W + w] != ref) bad++;
                    }
        printf("      NCHW * N C 1 1 -> mismatches=%d/%d\n", bad, N * C * H * W);
        chk("broadcast multiply matches reference", bad == 0);
    }

    printf("\n==== %s (%d failures) ====\n", g_fail ? "FAILURES" : "ALL PASS", g_fail);
    return g_fail ? 1 : 0;
}
