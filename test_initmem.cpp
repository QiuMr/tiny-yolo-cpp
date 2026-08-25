// verify InitModelFromMemory path works
#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cstdio>
#include <cstdlib>

typedef int (__stdcall *InitMem_t)(unsigned char*, int);
typedef int (__stdcall *Detect_t)(unsigned char*, int, float, float, float*, int);
typedef void (__stdcall *Release_t)();
typedef void (__stdcall *SetSize_t)(int, int);

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1] : "yolo11n.onnx";
    const char* image = argc > 2 ? argv[2] : "zidane.jpg";

#ifdef _WIN64
    HMODULE dll = LoadLibraryA("tiny_yolo.dll");
#else
    HMODULE dll = LoadLibraryA("tiny_yolo_x86.dll");
#endif
    if (!dll) { printf("load dll fail %d\n", GetLastError()); return 1; }
#ifdef _WIN64
    auto initm = (InitMem_t)GetProcAddress(dll, "InitModelFromMemory");
    auto detect = (Detect_t)GetProcAddress(dll, "YoloDetectFromMemory");
    auto rel = (Release_t)GetProcAddress(dll, "ReleaseModel");
    auto ss = (SetSize_t)GetProcAddress(dll, "SetInputSize");
#else
    auto initm = (InitMem_t)GetProcAddress(dll, "_InitModelFromMemory@8");
    auto detect = (Detect_t)GetProcAddress(dll, "_YoloDetectFromMemory@24");
    auto rel = (Release_t)GetProcAddress(dll, "_ReleaseModel@0");
    auto ss = (SetSize_t)GetProcAddress(dll, "_SetInputSize@8");
#endif

    FILE* f = fopen(model, "rb");
    if (!f) { printf("open model fail\n"); return 1; }
    fseek(f, 0, SEEK_END); int msz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* mbuf = (unsigned char*)malloc(msz);
    fread(mbuf, 1, msz, f); fclose(f);
    printf("model bytes: %d\n", msz);

    int r = initm(mbuf, msz);
    printf("InitModelFromMemory -> %d\n", r);
    if (r != 1) return 1;
    ss(640, 640);

    f = fopen(image, "rb");
    fseek(f, 0, SEEK_END); int isz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* ibuf = (unsigned char*)malloc(isz);
    fread(ibuf, 1, isz, f); fclose(f);

    float res[600];
    int n = 0;
    for (int run = 0; run < 3; run++) {
        LARGE_INTEGER t0, t1, fr; QueryPerformanceFrequency(&fr); QueryPerformanceCounter(&t0);
        n = detect(ibuf, isz, 0.25f, 0.45f, res, 100);
        QueryPerformanceCounter(&t1);
        printf("run%d: %.1fms count=%d\n", run,
               (t1.QuadPart - t0.QuadPart) * 1000.0 / fr.QuadPart, n);
    }
    for (int i = 0; i < n && i < 5; i++)
        printf("  [%d] cls=%.0f conf=%.3f box=(%.1f,%.1f,%.1f,%.1f)\n", i,
               res[i*6], res[i*6+1], res[i*6+2], res[i*6+3], res[i*6+4], res[i*6+5]);
    rel();
    return 0;
}
