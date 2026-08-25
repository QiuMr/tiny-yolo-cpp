#define _CRT_SECURE_NO_WARNINGS
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

typedef int (__stdcall *InitModel_t)(const char*);
typedef int (__stdcall *YoloDetectFromMemory_t)(unsigned char*, int, float, float, float*, int);
typedef void (__stdcall *ReleaseModel_t)();
typedef void (__stdcall *SetInputSize_t)(int, int);

int main(int argc, char* argv[]) {
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    if (argc < 3) {
        printf("Usage: test_tiny <model.tyro> <image.jpg> [input_size]\n");
        return 1;
    }

    int input_size = 640;
    if (argc >= 4) input_size = atoi(argv[3]);
    float conf_thres = 0.25f;
    if (argc >= 5) conf_thres = (float)atof(argv[4]);

    printf("Loading DLL...\n");
#ifdef _WIN64
    HMODULE dll = LoadLibraryA("tiny_yolo.dll");
#else
    HMODULE dll = LoadLibraryA("tiny_yolo_x86.dll");
#endif
    if (!dll) { printf("Load failed: %d\n", GetLastError()); return 1; }

    auto Init = (InitModel_t)GetProcAddress(dll,
#ifdef _WIN64
        "InitModel"
#else
        "_InitModel@4"
#endif
    );
    auto Detect = (YoloDetectFromMemory_t)GetProcAddress(dll,
#ifdef _WIN64
        "YoloDetectFromMemory"
#else
        "_YoloDetectFromMemory@24"
#endif
    );
    auto Release = (ReleaseModel_t)GetProcAddress(dll,
#ifdef _WIN64
        "ReleaseModel"
#else
        "_ReleaseModel@0"
#endif
    );
    auto SetInputSize = (SetInputSize_t)GetProcAddress(dll,
#ifdef _WIN64
        "SetInputSize"
#else
        "_SetInputSize@8"
#endif
    );

    if (!Init || !Detect || !Release) {
        printf("GetProcAddress failed\n");
        return 1;
    }

    if (SetInputSize) {
        SetInputSize(input_size, input_size);
        printf("Input size set to: %dx%d\n", input_size, input_size);
    }

    printf("InitModel...\n");
    int ret = Init(argv[1]);
    printf("InitModel returned: %d\n", ret);
    if (ret != 1) { printf("Init failed\n"); return 1; }

    printf("Reading image...\n");
    FILE* f = fopen(argv[2], "rb");
    if (!f) { printf("Open image failed\n"); return 1; }
    fseek(f, 0, SEEK_END); int isz = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char* id = (unsigned char*)malloc(isz);
    fread(id, 1, isz, f);
    fclose(f);
    printf("Image size: %d bytes\n", isz);

    float results[100 * 6];

    // 跑5次取最优
    for (int run = 0; run < 5; run++) {
        printf("Run %d... ", run);
        fflush(stdout);
        auto t0 = std::chrono::high_resolution_clock::now();
        int count = Detect(id, isz, conf_thres, 0.45f, results, 100);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("time=%.0f ms, count=%d\n", ms, count);

        if (run == 4) {
            for (int i = 0; i < count && i < 10; i++) {
                printf("  [%d] cls=%.0f conf=%.3f box=(%.1f,%.1f,%.1f,%.1f)\n",
                       i, results[i*6+0], results[i*6+1],
                       results[i*6+2], results[i*6+3], results[i*6+4], results[i*6+5]);
            }
        }
    }

    Release();
    FreeLibrary(dll);
    free(id);
    printf("\n=== Done ===\n");
    return 0;
}
