// Tiny YOLO - DLL 鍏ュ彛鍜?C 鎺ュ彛灏佽
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

// 鍙厤缃殑杈撳叆鍒嗚鲸鐜囷紙榛樿640锛屽彲閫氳繃SetInputSize淇敼锛?
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

// 璋冭瘯瀹?

// 鍔犺浇妯″瀷锛堜粠 .tyro 鏂囦欢璺緞锛?
extern "C" __declspec(dllexport) int __stdcall InitModel(const char* model_path) {
    if (g_model) return 1;
    if (!model_path) return -1;
    g_model = new TinyModel();

    // 鏍规嵁鏂囦欢鎵╁睍鍚嶅垽鏂牸寮?
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

// 浠庡唴瀛樺姞杞芥ā鍨嬫暟鎹紙鑷姩璇嗗埆 tyro / onnx 鏍煎紡锛?
extern "C" __declspec(dllexport) int __stdcall InitModelFromMemory(unsigned char* model_data, int model_size) {
    if (g_model) return 1;
    if (!model_data || model_size <= 0) return -1;
    // 鎸夐瓟鏁板垎娴侊細TYO1 = tyro 鏍煎紡锛涘惁鍒欏綋浣?ONNX (protobuf)
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

// 鎺ㄧ悊
extern "C" __declspec(dllexport) int __stdcall YoloDetectFromMemory(
    unsigned char* img_data, int img_size,
    float conf_thres, float nms_thres,
    float* results, int max_size) {

    if (!g_model || !img_data || img_size <= 0 || !results || max_size <= 0) return 0;

    try {
        // 1. 瑙ｇ爜鍥剧墖
        int w, h, c;
        unsigned char* img = stbi_load_from_memory(img_data, img_size, &w, &h, &c, 3);
        if (!img) { return 0; }
        // 璋冭瘯锛歍INY_YOLO_DUMP=1 鏃舵墦鍗板浘鐗囦笌 letterbox 鍙傛暟
        static const bool dbg_dump = (getenv("TINY_YOLO_DUMP") != nullptr);
        float scale = std::min((float)g_input_w / w, (float)g_input_h / h);
        int new_w = (int)(w * scale), new_h = (int)(h * scale);
        int pad_w = (g_input_w - new_w) / 2, pad_h = (g_input_h - new_h) / 2;
        if (dbg_dump)
            printf("[det] img=%dx%d scale=%.4f pad=%d,%d raw_box@3392 will follow\n",
                   w, h, scale, pad_w, pad_h);

        unsigned char* resized = (unsigned char*)malloc(new_w * new_h * 3);
        stbir_resize_uint8_linear(img, w, h, 0, resized, new_w, new_h, 0, (stbir_pixel_layout)3);
        stbi_image_free(img);

        // 3. 杞?NCHW float32 + 褰掍竴鍖栵紙letterbox 濉厖鍊?114 鐏帮級
        // 缂撳啿璺ㄥ抚澶嶇敤锛氳竟鐣屾亽涓?114 鐏帮紝棣栧抚濉ソ鍚庢棤闇€閲嶅～锛屼腑蹇冨尯鍩熸瘡甯ф暣浣撹鐩?
        static thread_local std::vector<float> tensor_data;
        size_t tensor_size = (size_t)3 * g_input_h * g_input_w;
        if (tensor_data.size() != tensor_size) tensor_data.assign(tensor_size, 114.0f / 255.0f);
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
        // 4. 璁剧疆杈撳叆骞舵墽琛?
        g_model->set_input(0, {1, 3, g_input_h, g_input_w}, tensor_data.data());
        if (!g_model->run()) { return 0; }
        // 5. 鑾峰彇杈撳嚭
        const Tensor* out = g_model->get_output(0);
        if (!out) { return 0; }

        // 妫€娴嬭緭鍑烘牸寮?
        bool is_v26_format = (out->ndim == 3 && out->shape[0] == 1 && out->shape[2] == 6);
        // v8/v10/v11/v12: [1, 4+nc, anchors]锛屾敮鎸佷换鎰忕被鍒暟锛堜笉闄愬畾 84锛?
        bool is_v8_format = (out->ndim == 3 && out->shape[0] == 1
                             && !is_v26_format && out->shape[1] > 4);

        // 杈撳嚭寮犻噺杞偍锛圱INY_YOLO_DUMP=1锛夛細鎵撳嵃褰㈢姸涓庢瘡琛屽墠鍑犱釜鍊硷紝璇婃柇鑷畾涔夋ā鍨嬪竷灞€
        if (dbg_dump) {
            printf("[fmt] ndim=%d shape=[", out->ndim);
            for (int d = 0; d < out->ndim; d++) printf("%d,", out->shape[d]);
            printf("] v26=%d v8=%d\n", (int)is_v26_format, (int)is_v8_format);
        }

        if (!is_v26_format && !is_v8_format) {
            return 0;
        }

        const float* out_data = out->data;
        // 杈撳嚭寮犻噺杞偍锛圱INY_YOLO_DUMP=1锛夛細鎵撳嵃褰㈢姸涓庢瘡琛屽墠鍑犱釜鍊硷紝璇婃柇鑷畾涔夋ā鍨嬪竷灞€
        if (dbg_dump) {
            printf("[out] ndim=%d shape=[", out->ndim);
            for (int d = 0; d < out->ndim; d++) printf("%d,", out->shape[d]);
            printf("] numel=%d\n", out->numel);
            int rows = out->ndim == 3 ? out->shape[1] : 6;
            int cols = out->ndim == 3 ? out->shape[2] : out->numel / 6;
            for (int r = 0; r < rows && r < 12; r++) {
                printf("  row[%d]:", r);
                for (int c = 0; c < 6 && c < cols; c++)
                    printf(" %.4g", out_data[(size_t)r * cols + c]);
                printf("\n");
            }
            // 鎵炬渶楂樺垎绫诲埆鍒嗗苟鎵撳嵃鍏舵鍊硷紙涓?ONNX Runtime 瀵圭収锛?
            if (out->ndim == 3 && rows > 4) {
                int classes = rows - 4;
                float best = 0; int bi = 0, bc = 0;
                for (int i = 0; i < cols; i++)
                    for (int cc = 0; cc < classes; cc++) {
                        float s = out_data[(size_t)(4 + cc) * cols + i];
                        if (s > best) { best = s; bi = i; bc = cc; }
                    }
                printf("  best: cls=%d score=%.4f at anchor %d -> box=[%.2f %.2f %.2f %.2f]\n",
                       bc, best, bi, out_data[bi], out_data[(size_t)cols + bi],
                       out_data[(size_t)2 * cols + bi], out_data[(size_t)3 * cols + bi]);
                printf("  anchor 3312 box=[%.4g %.4g %.4g %.4g]\n",
                       out_data[3312], out_data[(size_t)cols + 3312],
                       out_data[(size_t)2 * cols + 3312], out_data[(size_t)3 * cols + 3312]);
            }
        }
        std::vector<Box> boxes;

        if (is_v26_format) {
            // YOLOv26鏍煎紡锛歔1, num_dets, 6]锛屾瘡涓娴媅x1,y1,x2,y2,conf,cls]
            int num_dets = out->shape[1];
            for (int i = 0; i < num_dets; ++i) {
                float x1 = out_data[i * 6 + 0];
                float y1 = out_data[i * 6 + 1];
                float x2 = out_data[i * 6 + 2];
                float y2 = out_data[i * 6 + 3];
                float conf = out_data[i * 6 + 4];
                int cls = (int)out_data[i * 6 + 5];
                if (conf > conf_thres) {
                    // 杞崲涓哄乏涓婅+瀹介珮鏍煎紡锛屽苟搴旂敤letterbox閫嗗彉鎹?
                    float x = (x1 - pad_w) / scale;
                    float y = (y1 - pad_h) / scale;
                    float w = (x2 - x1) / scale;
                    float h = (y2 - y1) / scale;
                    boxes.push_back({x, y, w, h, conf, cls});
                }
            }
            // YOLOv26宸插唴缃甆MS锛岀洿鎺ヨ緭鍑?
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

        // YOLOv8/v10/v11/v12鏍煎紡锛歔1, 4+nc, num_anchors]
        int num_channels = out->shape[1];
        int num_anchors = out->shape[2];
        int classes = num_channels - 4;

        // 鍗曟鎵弿锛氭眰姣忎釜 anchor 鐨勬渶澶х被鍒垎骞剁紦瀛橈紙鍘熷疄鐜版壂涓ら亶锛岃法姝ヨ瀛樼炕鍊嶏級
        std::vector<float> best_score(num_anchors);
        std::vector<int> best_cls(num_anchors);
        int active_anchors = 0;
        int exact_zero_scores = 0;
        for (int i = 0; i < num_anchors; ++i) {
            float max_score = 0;
            int class_id = 0;
            for (int c = 0; c < classes; ++c) {
                float s = out_data[(4 + c) * num_anchors + i];
                if (s > max_score) { max_score = s; class_id = c; }
            }
            best_score[i] = max_score;
            best_cls[i] = class_id;
            if (max_score > 0.01f) active_anchors++;
            if (max_score == 0.0f) exact_zero_scores++;
        }

        // V10 鍒ゅ畾锛堝弻閲嶉獙璇侊紝闃茶鍒わ級锛?
        // 鐪熉穠10 绔埌绔鍑轰細鎶婃湭閫変腑 anchor 鐨勫垎鏁版帺鐮佷负绮剧‘ 0锛堢█鐤忕巼 >90%锛夛紱
        // 鑰屾櫘閫?v8/v12 sigmoid 鍒嗘暟鍑犱箮涓嶄細绮剧‘涓?0銆備粎鏁伴噺闃堝€间細璇激
        // 绫诲埆灏戙€佽儗鏅姂鍒跺己鐨勮嚜瀹氫箟妯″瀷锛堝 4 绫绘ā鍨嬪彧鏈?<100 涓?anchor 杩?0.01锛夈€?
        bool is_v10_format = (active_anchors < 100) && (num_anchors > 0)
                             && (exact_zero_scores > num_anchors * 9 / 10);
        if (is_v10_format) {
            // 鍑犱綍浜ゅ弶楠岃瘉锛歷10 鐨?box 鏄?xyxy锛岄渶婊¤冻 x2>x1 涓?y2>y1锛?
            // 鍙栧垎鏈€楂樼殑鍑犱釜 anchor 妫€鏌ワ紝鑻ヤ笉婊¤冻鍒欎粛鎸?v8 澶勭悊
            int checked = 0, geo_ok = 0;
            for (int i = 0; i < num_anchors && checked < 8; ++i) {
                if (best_score[i] <= 0.5f) continue;
                float b0 = out_data[0 * num_anchors + i];
                float b1 = out_data[1 * num_anchors + i];
                float b2 = out_data[2 * num_anchors + i];
                float b3 = out_data[3 * num_anchors + i];
                if (b2 > b0 && b3 > b1) geo_ok++;
                checked++;
            }
            if (checked > 0 && geo_ok < checked) is_v10_format = false;
        }

        // 鍚庡鐞嗭細瑙ｆ瀽妫€娴嬫
        for (int i = 0; i < num_anchors; ++i) {
            float max_score = best_score[i];
            if (max_score > conf_thres) {
                float b0 = out_data[0 * num_anchors + i];
                float b1 = out_data[1 * num_anchors + i];
                float b2 = out_data[2 * num_anchors + i];
                float b3 = out_data[3 * num_anchors + i];

                float cx, cy, bw, bh;
                if (is_v10_format) {
                    // V10: box鏄?x1, y1, x2, y2)鏍煎紡锛岃浆鎹负(cx, cy, w, h)
                    cx = (b0 + b2) * 0.5f;
                    cy = (b1 + b3) * 0.5f;
                    bw = b2 - b0;
                    bh = b3 - b1;
                } else {
                    // V8/V11/V12: box鏄?cx, cy, w, h)鏍煎紡
                    cx = b0;
                    cy = b1;
                    bw = b2;
                    bh = b3;
                }

                boxes.push_back({
                    (cx - pad_w) / scale, (cy - pad_h) / scale,
                    bw / scale, bh / scale,
                    max_score, best_cls[i]
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

BOOL APIENTRY DllMain(HMODULE h, DWORD r, LPVOID l) {
    if (r == DLL_PROCESS_DETACH) {
        // 鏄撹瑷€ IDE/缂栬瘧鍚庣殑 exe 閫€鍑烘椂锛岃嫢鐢ㄦ埛鏈樉寮忚皟鐢?ReleaseModel锛岃繖閲屽仛鍏滃簳娓呯悊
        // l != NULL 琛ㄧず杩涚▼姝ｅ湪缁堟锛圗xitProcess锛夛紝姝ゆ椂涓嶈兘闃诲绛夊緟绾跨▼ join锛屽惁鍒欎細姝婚攣 loader lock
        // 閲囩敤 detach 鏂瑰紡璁╃郴缁熺洿鎺ュ洖鏀?
        if (l != NULL) {
            // 杩涚▼缁堟锛氬揩閫?detach锛屼笉绛夊緟
            if (g_model) { g_model = nullptr; /* 娉勬紡涓€鐐瑰唴瀛樼敱绯荤粺鍥炴敹锛岄伩鍏嶆瀽鏋勯樆濉?*/ }
            SimpleThreadPool::instance().shutdown_detach();
        } else {
            // FreeLibrary 涓诲姩鍗歌浇锛氬彲浠ュ畨鍏?join
            if (g_model) { delete g_model; g_model = nullptr; }
            else SimpleThreadPool::instance().shutdown();
        }
    }
    return TRUE;
}
