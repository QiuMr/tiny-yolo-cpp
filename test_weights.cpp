// scan weights of loaded onnx model for NaN/Inf
#define _CRT_SECURE_NO_WARNINGS
#include "tiny_yolo/model.h"
#include <cstdio>

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "yolo12n.onnx";
    TinyModel m;
    if (!m.load_from_onnx(path)) { printf("load failed\n"); return 1; }
    size_t total_nan = 0, total_inf = 0, bad_tensors = 0;
    for (size_t i = 0; i < m.tensor_infos.size(); i++) {
        TensorInfo& ti = m.tensor_infos[i];
        Tensor& t = m.tensors[i];
        if (!ti.is_const || !t.data) continue;
        if (ti.dtype != 0 && ti.dtype != 1) continue; // float only
        if (t.numel <= 0) continue;
        size_t nan_cnt = 0, inf_cnt = 0;
        float mn = 1e30f, mx = -1e30f;
        for (int k = 0; k < t.numel; k++) {
            float v = t.data[k];
            if (_isnan(v)) { nan_cnt++; continue; }
            if (!_finite(v)) { inf_cnt++; continue; }
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (nan_cnt || inf_cnt) {
            bad_tensors++;
            total_nan += nan_cnt; total_inf += inf_cnt;
            printf("tensor %u ndim=%d shape=[", ti.id, ti.ndim);
            for (int d = 0; d < ti.ndim; d++) printf("%d,", ti.shape[d]);
            printf("] numel=%d nan=%zu inf=%zu range=[%g,%g]\n", t.numel, nan_cnt, inf_cnt,
                   bad_tensors ? mn : 0.0f, mx);
        }
    }
    printf("done: %zu bad tensors, nan=%zu inf=%zu\n", bad_tensors, total_nan, total_inf);
    return 0;
}
