// conv3x3s1_winograd unit test: F(2,3) vs F(4,3) against naive reference
#include "tiny_yolo/ops.h"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

int main() {
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-1.5f, 1.5f);
    int cases[][4] = {
        {3, 16, 8, 8}, {16, 32, 13, 11}, {32, 64, 32, 32}, {64, 128, 20, 20},
        {64, 64, 80, 80}, {128, 128, 40, 40}, {256, 128, 10, 10}, {7, 5, 9, 9},
    };
    bool all_ok = true;
    for (auto& cs : cases) {
        int Cin = cs[0], Cout = cs[1], H = cs[2], W = cs[3];
        std::vector<float> in((size_t)Cin * H * W), w((size_t)Cout * Cin * 9), b(Cout);
        for (auto& v : in) v = dist(rng);
        for (auto& v : w) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        // naive reference, 3x3 s1 p1
        std::vector<float> ref((size_t)Cout * H * W);
        for (int silu_i = 0; silu_i <= 1; silu_i++) {
            bool silu = silu_i != 0;
            for (int p = 0; p < Cout; p++)
                for (int oh = 0; oh < H; oh++)
                    for (int ow = 0; ow < W; ow++) {
                        float s = b[p];
                        for (int q = 0; q < Cin; q++)
                            for (int kh = 0; kh < 3; kh++)
                                for (int kw = 0; kw < 3; kw++) {
                                    int ih = oh - 1 + kh, iw = ow - 1 + kw;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        s += in[((size_t)q * H + ih) * W + iw] * w[((size_t)p * Cin + q) * 9 + kh * 3 + kw];
                                }
                        if (silu) s = s / (1.0f + fast_exp(-s));
                        ref[((size_t)p * H + oh) * W + ow] = s;
                    }
            for (int mode = 1; mode <= 2; mode++) {
                std::vector<float> out((size_t)Cout * H * W);
                memset(out.data(), 0, out.size() * sizeof(float));
                conv3x3s1_winograd(in.data(), w.data(), b.data(), out.data(), silu,
                                   1, Cin, H, W, Cout, mode);
                double maxerr = 0; size_t bad = 0;
                for (size_t i = 0; i < ref.size(); i++) {
                    double e = fabs((double)ref[i] - out[i]);
                    if (e > maxerr) maxerr = e;
                    if (e > 2e-3 * (2.0 + fabs(ref[i]))) bad++;
                }
                printf("Cin=%3d Cout=%3d H=%3d W=%3d silu=%d F(%d,3): maxerr=%.6g bad=%zu\n",
                       Cin, Cout, H, W, silu_i, mode == 2 ? 4 : 2, maxerr, bad);
                if (bad) all_ok = false;
            }
        }
    }
    printf(all_ok ? "ALL PASS\n" : "FAILED\n");
    return all_ok ? 0 : 1;
}
