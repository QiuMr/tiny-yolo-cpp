// conv3x3s2p1 unit test: compare against naive reference
#include "tiny_yolo/ops.h"
#include <cstdio>
#include <random>
#include <vector>
#include <cmath>

int main() {
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-2.f, 2.f);
    int cases[][4] = {
        {3, 5, 11, 13}, {16, 32, 32, 32}, {64, 64, 20, 20},
        {5, 4, 9, 10}, {32, 48, 160, 160}, {8, 8, 7, 7},
    };
    bool all_ok = true;
    for (auto& cs : cases) {
        int Cin = cs[0], Cout = cs[1], H = cs[2], W = cs[3];
        int oH = (H + 2 - 3) / 2 + 1, oW = (W + 2 - 3) / 2 + 1;
        std::vector<float> in((size_t)Cin * H * W), w((size_t)Cout * Cin * 9), b(Cout);
        std::vector<float> out((size_t)Cout * oH * oW), ref((size_t)Cout * oH * oW);
        for (auto& v : in) v = dist(rng);
        for (auto& v : w) v = dist(rng);
        for (auto& v : b) v = dist(rng);

        for (int silu_i = 0; silu_i <= 1; silu_i++) {
            bool silu = silu_i != 0;
            // naive reference
            for (int p = 0; p < Cout; p++)
                for (int oh = 0; oh < oH; oh++)
                    for (int ow = 0; ow < oW; ow++) {
                        float s = b[p];
                        for (int q = 0; q < Cin; q++)
                            for (int kh = 0; kh < 3; kh++)
                                for (int kw = 0; kw < 3; kw++) {
                                    int ih = oh * 2 - 1 + kh, iw = ow * 2 - 1 + kw;
                                    if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                        s += in[((size_t)q * H + ih) * W + iw] * w[((size_t)p * Cin + q) * 9 + kh * 3 + kw];
                                }
                        if (silu) s = s / (1.0f + fast_exp(-s));
                        ref[((size_t)p * oH + oh) * oW + ow] = s;
                    }
            memset(out.data(), 0, out.size() * sizeof(float));
            conv3x3s2p1(in.data(), w.data(), b.data(), out.data(), 1, Cin, H, W, Cout, silu);
            double maxerr = 0; size_t bad = 0;
            for (size_t i = 0; i < ref.size(); i++) {
                double e = fabs((double)ref[i] - out[i]);
                if (e > maxerr) maxerr = e;
                if (e > 1e-3 * (1.0 + fabs(ref[i]))) bad++;
            }
            printf("Cin=%3d Cout=%3d H=%3d W=%3d silu=%d -> %dx%d  maxerr=%.6g  bad=%zu\n",
                   Cin, Cout, H, W, silu_i, oH, oW, maxerr, bad);
            if (bad) all_ok = false;
        }
    }
    printf(all_ok ? "ALL PASS\n" : "FAILED\n");
    return all_ok ? 0 : 1;
}
