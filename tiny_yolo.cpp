// Tiny YOLO - DLL 入口和 C 接口封装
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define NDEBUG
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <vector>
#include <cstdio>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb_image_resize2.h"

#include "tiny_yolo/model.h"

static TinyModel* g_model = nullptr;

// 可配置的输入分辨率（默认640，可通过SetInputSize修改）
static int g_input_w = 640;
static int g_input_h = 640;

extern "C" __declspec(dllexport) void __stdcall SetInputSize(int w, int h) {
    g_input_w = w;
    g_input_h = h;
}

struct Box { float x, y, w, h, score; int label; };

static inline float iou(const Box& a, const Box& b) {
    float x1 = std::max(a.x - a.w * 0.5f, b.x - b.w * 0.5f);
    float y1 = std::max(a.y - a.h * 0.5f, b.y - b.h * 0.5f);
    float x2 = std::min(a.x + a.w * 0.5f, b.x + b.w * 0.5f);
    float y2 = std::min(a.y + a.h * 0.5f, b.y + b.h * 0.5f);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    float interArea = (x2 - x1) * (y2 - y1);
    float unionArea = a.w * a.h + b.w * b.h - interArea;
    return interArea / unionArea;
}

// 调试宏

// 加载模型（从 .tyro 文件路径）
extern "C" __declspec(dllexport) int __stdcall InitModel(const char* model_path) {
    if (g_model) return 1;
    if (!model_path) return -1;
    g_model = new TinyModel();

    // 根据文件扩展名判断格式
    const char* ext = strrchr(model_path, '.');
    bool is_onnx = (ext && (_stricmp(ext, ".onnx") == 0));

    bool ok;
    if (is_onnx) {
        ok = g_model->load_from_onnx(model_path);
    } else {
        ok = g_model->load_from_file(model_path);
    }

    if (!ok) {
        delete g_model; g_model = nullptr; return -2;
    }
    return 1;
}

// 从内存加载模型数据（自动识别 tyro / onnx 格式）
extern "C" __declspec(dllexport) int __stdcall InitModelFromMemory(unsigned char* model_data, int model_size) {
    if (g_model) return 1;
    if (!model_data || model_size <= 0) return -1;
    // 按魔数分流：TYO1 = tyro 格式；否则当作 ONNX (protobuf)
    bool is_tyro = (model_size >= 4 && *(const uint32_t*)model_data == TYO_MAGIC);
    char tempFile[MAX_PATH];
    GetTempPathA(MAX_PATH, tempFile);
    strcat_s(tempFile, is_tyro ? "tiny_yolo_model.tyro" : "tiny_yolo_model.onnx");
    FILE* f = fopen(tempFile, "wb");
    if (!f) return -2;
    fwrite(model_data, 1, model_size, f);
    fclose(f);
    g_model = new TinyModel();
    bool ok = is_tyro ? g_model->load_from_file(tempFile)
                      : g_model->load_from_onnx(tempFile);
    if (!ok) {
        delete g_model; g_model = nullptr; return -3;
    }
    return 1;
}

// 推理
extern "C" __declspec(dllexport) int __stdcall YoloDetectFromMemory(
    unsigned char* img_data, int img_size,
    float conf_thres, float nms_thres,
    float* results, int max_size) {

    if (!g_model || !img_data || img_size <= 0 || !results || max_size <= 0) return 0;

    try {
        // 1. 解码图片
        int w, h, c;
        unsigned char* img = stbi_load_from_memory(img_data, img_size, &w, &h, &c, 3);
        if (!img) { return 0; }
        // 2. Resize + Pad (letterbox)
        float scale = std::min((float)g_input_w / w, (float)g_input_h / h);
        int new_w = (int)(w * scale), new_h = (int)(h * scale);
        int pad_w = (g_input_w - new_w) / 2, pad_h = (g_input_h - new_h) / 2;

        unsigned char* resized = (unsigned char*)malloc(new_w * new_h * 3);
        stbir_resize_uint8_linear(img, w, h, 0, resized, new_w, new_h, 0, (stbir_pixel_layout)3);
        stbi_image_free(img);

        // 3. 转 NCHW float32 + 归一化（letterbox 填充值 114 灰）
        int tensor_size = 1 * 3 * g_input_h * g_input_w;
        std::vector<float> tensor_data(tensor_size, 114.0f / 255.0f);
        float* ptr_r = tensor_data.data();
        float* ptr_g = ptr_r + g_input_w * g_input_h;
        float* ptr_b = ptr_g + g_input_w * g_input_h;

        for (int y = 0; y < new_h; ++y) {
            for (int x = 0; x < new_w; ++x) {
                int dst = (y + pad_h) * g_input_w + (x + pad_w);
                int src = (y * new_w + x) * 3;
                ptr_r[dst] = resized[src + 0] / 255.0f;
                ptr_g[dst] = resized[src + 1] / 255.0f;
                ptr_b[dst] = resized[src + 2] / 255.0f;
            }
        }
        free(resized);
        // 4. 设置输入并执行
        g_model->set_input(0, {1, 3, g_input_h, g_input_w}, tensor_data.data());
        if (!g_model->run()) { return 0; }
        // 5. 获取输出
        const Tensor* out = g_model->get_output(0);
        if (!out) { return 0; }

        // 检测输出格式
        bool is_v26_format = (out->ndim == 3 && out->shape[0] == 1 && out->shape[2] == 6);
        // v8/v10/v11/v12: [1, 4+nc, anchors]，支持任意类别数（不限定 84）
        bool is_v8_format = (out->ndim == 3 && out->shape[0] == 1
                             && !is_v26_format && out->shape[1] > 4);

        if (!is_v26_format && !is_v8_format) {
            return 0;
        }

        const float* out_data = out->data;
        std::vector<Box> boxes;

        if (is_v26_format) {
            // YOLOv26格式：[1, num_dets, 6]，每个检测[x1,y1,x2,y2,conf,cls]
            int num_dets = out->shape[1];
            for (int i = 0; i < num_dets; ++i) {
                float x1 = out_data[i * 6 + 0];
                float y1 = out_data[i * 6 + 1];
                float x2 = out_data[i * 6 + 2];
                float y2 = out_data[i * 6 + 3];
                float conf = out_data[i * 6 + 4];
                int cls = (int)out_data[i * 6 + 5];
                if (conf > conf_thres) {
                    // 转换为左上角+宽高格式，并应用letterbox逆变换
                    float x = (x1 - pad_w) / scale;
                    float y = (y1 - pad_h) / scale;
                    float w = (x2 - x1) / scale;
                    float h = (y2 - y1) / scale;
                    boxes.push_back({x, y, w, h, conf, cls});
                }
            }
            // YOLOv26已内置NMS，直接输出
            int valid_count = 0;
            for (size_t i = 0; i < boxes.size() && valid_count < max_size; ++i) {
                results[valid_count * 6 + 0] = (float)boxes[i].label;
                results[valid_count * 6 + 1] = boxes[i].score;
                results[valid_count * 6 + 2] = boxes[i].x;
                results[valid_count * 6 + 3] = boxes[i].y;
                results[valid_count * 6 + 4] = boxes[i].w;
                results[valid_count * 6 + 5] = boxes[i].h;
                valid_count++;
            }
            return valid_count;
        }

        // YOLOv8/v10/v11/v12格式：[1, 84, num_anchors]
        int num_channels = out->shape[1];
        int num_anchors = out->shape[2];
        int classes = num_channels - 4;

        // 自动检测V10：V10是end-to-end检测，只有少数anchor有非零输出
        // V8/V11/V12的8400个anchor中很多都有非零类别分数
        int active_anchors = 0;
        for (int i = 0; i < num_anchors; ++i) {
            float max_score = 0;
            for (int c = 0; c < classes; ++c) {
                float s = out_data[(4 + c) * num_anchors + i];
                if (s > max_score) max_score = s;
            }
            if (max_score > 0.01f) active_anchors++;
        }
        bool is_v10_format = (active_anchors < 100); // V10通常只有几十个active anchor，V8有几百上千个

        // 6. 后处理：解析检测框
        for (int i = 0; i < num_anchors; ++i) {
            float max_score = 0;
            int class_id = 0;
            for (int c = 0; c < classes; ++c) {
                float s = out_data[(4 + c) * num_anchors + i];
                if (s > max_score) { max_score = s; class_id = c; }
            }
            if (max_score > conf_thres) {
                float b0 = out_data[0 * num_anchors + i];
                float b1 = out_data[1 * num_anchors + i];
                float b2 = out_data[2 * num_anchors + i];
                float b3 = out_data[3 * num_anchors + i];

                float cx, cy, bw, bh;
                if (is_v10_format) {
                    // V10: box是(x1, y1, x2, y2)格式，转换为(cx, cy, w, h)
                    cx = (b0 + b2) * 0.5f;
                    cy = (b1 + b3) * 0.5f;
                    bw = b2 - b0;
                    bh = b3 - b1;
                } else {
                    // V8/V11/V12: box是(cx, cy, w, h)格式
                    cx = b0;
                    cy = b1;
                    bw = b2;
                    bh = b3;
                }

                boxes.push_back({
                    (cx - pad_w) / scale, (cy - pad_h) / scale,
                    bw / scale, bh / scale,
                    max_score, class_id
                });
            }
        }

        // 7. NMS
        std::sort(boxes.begin(), boxes.end(), [](const Box& a, const Box& b) { return a.score > b.score; });

        int valid_count = 0;
        std::vector<bool> sup(boxes.size(), false);
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (sup[i]) continue;
            if (valid_count >= max_size) break;
            results[valid_count * 6 + 0] = (float)boxes[i].label;
            results[valid_count * 6 + 1] = boxes[i].score;
            results[valid_count * 6 + 2] = boxes[i].x;
            results[valid_count * 6 + 3] = boxes[i].y;
            results[valid_count * 6 + 4] = boxes[i].w;
            results[valid_count * 6 + 5] = boxes[i].h;
            valid_count++;

            for (size_t j = i + 1; j < boxes.size(); ++j)
                if (!sup[j] && boxes[j].label == boxes[i].label && iou(boxes[i], boxes[j]) > nms_thres) sup[j] = true;
        }
        return valid_count;
    } catch (...) { return 0; }
}

extern "C" __declspec(dllexport) void __stdcall ReleaseModel() {
    if (g_model) { delete g_model; g_model = nullptr; }
}

BOOL APIENTRY DllMain(HMODULE h, DWORD r, LPVOID l) { return TRUE; }
