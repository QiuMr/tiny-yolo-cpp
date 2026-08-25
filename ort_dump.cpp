// ort_dump: run model via ONNX Runtime, dump output tensor layout/values as ground truth
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#include <windows.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

#include "include/onnxruntime_c_api.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : "yolo12n.onnx";
    const char* image = argc > 2 ? argv[2] : "test.jpg";
    int input_size = argc > 3 ? atoi(argv[3]) : 640;

    // dynamic load
    HMODULE dll = LoadLibraryA("onnxruntime.dll");
    if (!dll) { printf("no onnxruntime.dll\n"); return 1; }
    auto get_base = (const OrtApiBase* (*)(void))GetProcAddress(dll, "OrtGetApiBase");
    const OrtApi* ort = get_base()->GetApi(17);
    if (!ort) return 1;

    OrtEnv* env = nullptr;
    ort->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "d", &env);
    OrtSessionOptions* so = nullptr;
    ort->CreateSessionOptions(&so);
    ort->SetIntraOpNumThreads(so, 8);
    OrtSession* session = nullptr;
    wchar_t wmodel[MAX_PATH];
    MultiByteToWideChar(CP_UTF8, 0, model, -1, wmodel, MAX_PATH);
    OrtStatus* st = ort->CreateSession(env, wmodel, so, &session);
    if (st) { printf("session err: %s\n", ort->GetErrorMessage(st)); return 1; }

    // preprocess: same letterbox as tiny_yolo
    int w, h, c;
    unsigned char* img = stbi_load(image, &w, &h, &c, 3);
    if (!img) { printf("img fail\n"); return 1; }
    float scale = std::min((float)input_size / w, (float)input_size / h);
    int nw = (int)(w * scale), nh = (int)(h * scale);
    int pw = (input_size - nw) / 2, ph = (input_size - nh) / 2;
    std::vector<unsigned char> resized((size_t)nw * nh * 3);
    stbir_resize_uint8_linear(img, w, h, 0, resized.data(), nw, nh, 0, (stbir_pixel_layout)3);
    std::vector<float> tensor((size_t)3 * input_size * input_size, 114.0f / 255.0f);
    float* pr = tensor.data();
    float* pg = pr + input_size * input_size;
    float* pb = pg + input_size * input_size;
    for (int y = 0; y < nh; y++)
        for (int x = 0; x < nw; x++) {
            int dst = (y + ph) * input_size + (x + pw);
            int src = (y * nw + x) * 3;
            pr[dst] = resized[src + 0] / 255.0f;
            pg[dst] = resized[src + 1] / 255.0f;
            pb[dst] = resized[src + 2] / 255.0f;
        }

    OrtMemoryInfo* minfo = nullptr;
    ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &minfo);
    const char* in_names[] = { "images" };
    const char* out_names[] = { "output0" };
    // use session input/output names properly
    OrtAllocator* alloc = nullptr;
    ort->GetAllocatorWithDefaultOptions(&alloc);
    char* in_name = nullptr; char* out_name = nullptr;
    ort->SessionGetInputName(session, 0, alloc, &in_name);
    ort->SessionGetOutputName(session, 0, alloc, &out_name);
    in_names[0] = in_name; out_names[0] = out_name;
    printf("input name=%s output name=%s\n", in_name, out_name);

    int64_t dims[4] = { 1, 3, input_size, input_size };
    OrtValue* tin = nullptr;
    ort->CreateTensorWithDataAsOrtValue(minfo, tensor.data(),
        tensor.size() * sizeof(float), dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &tin);

    OrtValue* tout = nullptr;
    st = ort->Run(session, nullptr, in_names, (const OrtValue* const*)&tin, 1, out_names, 1, &tout);
    if (st) { printf("run err: %s\n", ort->GetErrorMessage(st)); return 1; }

    float* odata = nullptr;
    OrtTensorTypeAndShapeInfo* tsi = nullptr;
    ort->GetTensorTypeAndShape(tout, &tsi);
    size_t ndim = 0;
    ort->GetDimensionsCount(tsi, &ndim);
    std::vector<int64_t> shape(ndim);
    ort->GetDimensions(tsi, shape.data(), ndim);
    size_t numel = 1;
    for (auto d : shape) numel *= d;
    ort->GetTensorMutableData(tout, (void**)&odata);

    printf("[ort] ndim=%zu shape=[", ndim);
    for (auto d : shape) printf("%lld,", (long long)d);
    printf("]\n");
    if (ndim == 3) {
        int rows = (int)shape[1], cols = (int)shape[2];
        for (int r = 0; r < rows && r < 12; r++) {
            printf("  row[%d]:", r);
            for (int c2 = 0; c2 < 6 && c2 < cols; c2++) printf(" %.4g", odata[(size_t)r * cols + c2]);
            printf("\n");
        }
        // find best score anchor
        int classes = rows - 4;
        float best = 0; int bi = 0, bc = 0;
        for (int i = 0; i < cols; i++)
            for (int cc = 0; cc < classes; cc++) {
                float s = odata[(size_t)(4 + cc) * cols + i];
                if (s > best) { best = s; bi = i; bc = cc; }
            }
        printf("  best: cls=%d score=%.4f at anchor %d -> box=[%.2f %.2f %.2f %.2f]\n",
               bc, best, bi, odata[bi], odata[(size_t)cols + bi],
               odata[(size_t)2 * cols + bi], odata[(size_t)3 * cols + bi]);
    }
    return 0;
}
