// Regression tests for the defects found in the 2026-09-08 review of tiny_yolo.
// Each test below failed (or silently misbehaved) before the fix.
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

typedef int  (__stdcall *InitModel_t)(const char*);
typedef int  (__stdcall *InitModelMem_t)(unsigned char*, int);
typedef int  (__stdcall *Detect_t)(unsigned char*, int, float, float, float*, int);
typedef void (__stdcall *ReleaseModel_t)();
typedef void (__stdcall *SetInputSize_t)(int, int);

static InitModel_t    Init;
static InitModelMem_t InitMem;
static Detect_t       Detect;
static ReleaseModel_t Release;
static SetInputSize_t SetSize;

static int g_pass = 0, g_fail = 0;

static void check(const char* name, bool ok) {
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name);
    if (ok) g_pass++; else g_fail++;
}

static unsigned char* read_file(const char* path, int* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("  cannot open %s\n", path); return 0; }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* p = (unsigned char*)malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { free(p); fclose(f); return 0; }
    fclose(f);
    *out_size = (int)n;
    return p;
}

#define MAXBOX 100
static float g_r1[MAXBOX * 6], g_r2[MAXBOX * 6], g_r3[MAXBOX * 6];

static int detect(unsigned char* img, int isz, float conf, float nms, float* res) {
    return Detect(img, isz, conf, nms, res, MAXBOX);
}

int main() {
    setvbuf(stdout, NULL, _IONBF, 0);

    HMODULE dll = LoadLibraryA("tiny_yolo_x86.dll");
    if (!dll) { printf("LoadLibrary failed\n"); return 1; }
    Init    = (InitModel_t)GetProcAddress(dll, "_InitModel@4");
    InitMem = (InitModelMem_t)GetProcAddress(dll, "_InitModelFromMemory@8");
    Detect  = (Detect_t)GetProcAddress(dll, "_YoloDetectFromMemory@24");
    Release = (ReleaseModel_t)GetProcAddress(dll, "_ReleaseModel@0");
    SetSize = (SetInputSize_t)GetProcAddress(dll, "_SetInputSize@8");
    if (!Init || !InitMem || !Detect || !Release || !SetSize) {
        printf("GetProcAddress failed\n"); return 1;
    }

    int isz = 0, tsz = 0, msz = 0;
    unsigned char* img  = read_file("1.jpg", &isz);
    unsigned char* tall = read_file("tall.jpg", &tsz);
    unsigned char* mdl  = read_file("yolo11n.onnx", &msz);
    if (!img || !tall || !mdl) return 1;

    // ------------------------------------------------------------------
    // T1: ReleaseModel() must reset the explicit input-size flag.
    // Before: SetInputSize(640) -> Release -> Init(m256, declares 256) left the
    // size at 640, and the guard silently returned 0 detections forever.
    printf("\nT1  ReleaseModel resets explicit input size\n");
    check("InitModel(yolo11n)", Init("yolo11n.onnx") == 1);
    SetSize(640, 640);
    int c1 = detect(img, isz, 0.25f, 0.45f, g_r1);
    printf("      yolo11n@640 -> %d box(es)\n", c1);
    check("yolo11n detects", c1 > 0);
    Release();
    check("InitModel(m256)", Init("m256.onnx") == 1);
    int c2 = detect(img, isz, 0.25f, 0.45f, g_r2);
    printf("      m256 (auto-adopted size) -> %d box(es), cls=%.0f conf=%.2f\n",
           c2, c2 > 0 ? g_r2[0] : -1.0f, c2 > 0 ? g_r2[1] : 0.0f);
    check("m256 not blocked by stale explicit size", c2 == 1);
    Release();

    // ------------------------------------------------------------------
    // T2: InitModelFromMemory must parse in place, not via a %TEMP% file
    // (which was also never deleted and used a fixed, colliding name).
    printf("\nT2  InitModelFromMemory is a true in-memory load\n");
    char tmp[MAX_PATH];
    GetTempPathA(MAX_PATH, tmp);
    strcat(tmp, "tiny_yolo_model.onnx");
    DeleteFileA(tmp);
    int rc = InitMem(mdl, msz);
    check("InitModelFromMemory returns 1", rc == 1);
    int c3 = detect(img, isz, 0.25f, 0.45f, g_r1);
    printf("      in-memory yolo11n -> %d box(es)\n", c3);
    check("in-memory model detects", c3 == c1 && c3 > 0);
    check("no temp model file written to %TEMP%",
          GetFileAttributesA(tmp) == INVALID_FILE_ATTRIBUTES);
    Release();

    // ------------------------------------------------------------------
    // T3: a second InitModel() without Release must actually swap models.
    // Before: it returned 1 while keeping yolo11n loaded.
    printf("\nT3  repeated InitModel swaps the model\n");
    check("InitModel(yolo11n)", Init("yolo11n.onnx") == 1);
    int c4 = detect(img, isz, 0.25f, 0.45f, g_r1);
    check("InitModel(m256) without release", Init("m256.onnx") == 1);
    int c5 = detect(img, isz, 0.25f, 0.45f, g_r2);
    printf("      before swap: %d box(es) cls=%.0f | after: %d box(es) cls=%.0f\n",
           c4, c4 > 0 ? g_r1[0] : -1.0f, c5, c5 > 0 ? g_r2[0] : -1.0f);
    check("second load really uses the new model (cls 4, not person cls 0)",
          c5 == 1 && (int)g_r2[0] == 4);
    Release();

    // ------------------------------------------------------------------
    // T4: letterbox padding must not carry pixels from the previous frame.
    // Before: the buffer was only primed with 114-gray on size change, so a
    // different aspect ratio left stale pixels outside the new image window.
    printf("\nT4  letterbox padding is not polluted by the previous frame\n");
    check("InitModel(yolo11n)", Init("yolo11n.onnx") == 1);
    SetSize(640, 640);
    int a1 = detect(img, isz, 0.25f, 0.45f, g_r1);
    int tb = detect(tall, tsz, 0.25f, 0.45f, g_r3);   // changes the geometry
    int a2 = detect(img, isz, 0.25f, 0.45f, g_r2);    // back to the first image
    printf("      1.jpg -> %d, tall.jpg -> %d, 1.jpg again -> %d\n", a1, tb, a2);
    bool same = (a1 == a2) && (memcmp(g_r1, g_r2, sizeof(float) * 6 * (a1 > 0 ? a1 : 0)) == 0);
    check("same image gives identical results across an aspect-ratio change", same);
    Release();

    // ------------------------------------------------------------------
    // T5: zero thresholds must be defaulted, not fed to an O(n^2) NMS.
    printf("\nT5  zero/NaN thresholds fall back to defaults\n");
    check("InitModel(yolo11n)", Init("yolo11n.onnx") == 1);
    SetSize(640, 640);
    int z = detect(img, isz, 0.0f, 0.0f, g_r1);
    printf("      conf=0 nms=0 -> %d box(es)\n", z);
    check("conf=0 does not hang and stays within max_size", z >= 0 && z <= MAXBOX);
    Release();

    // ------------------------------------------------------------------
    // T6: cross-model sanity (all must produce output, none may crash).
    printf("\nT6  cross-model sanity\n");
    const char* models[] = { "yolov8n.onnx", "yolov10n.onnx", "yolo26n.onnx" };
    for (int i = 0; i < 3; i++) {
        if (Init(models[i]) != 1) { check(models[i], false); continue; }
        SetSize(640, 640);
        int c = detect(img, isz, 0.25f, 0.45f, g_r1);
        printf("      %-14s -> %d box(es)\n", models[i], c);
        check(models[i], c > 0);
        Release();
    }

    printf("\n==== %d passed, %d failed ====\n", g_pass, g_fail);
    free(img); free(tall); free(mdl);
    FreeLibrary(dll);
    return g_fail ? 1 : 0;
}
