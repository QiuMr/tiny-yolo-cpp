# Tiny YOLO

手写 C++ YOLO 推理引擎，零依赖，AVX2 加速，直接编译为 32 位 DLL 供易语言调用。

> 纯 C++ 实现全部算子，无 OpenCV / ONNX Runtime 依赖。支持 ONNX 直载，兼容 YOLOv8 / v10 / v11 / v12 / v26。

![C++17](https://img.shields.io/badge/C++-17-00599C) ![Win32](https://img.shields.io/badge/Win32-x86%20%7C%20x64-555) ![License](https://img.shields.io/badge/license-MIT-green)

---

## 特性

- **零依赖** - 仅 `stb_image` + Win32 API
- **ONNX 直载** - `InitModel("yolo11n.onnx")` 即可，自动融合 `Conv+SiLU`
- **易语言友好** - `__stdcall` 导出，32 位 DLL，开箱即用
- **全系支持** - v8 / v10 / v11 / v12 / v26 自动识别，任意类别数
- **高性能** - AVX2 + Winograd + 事件驱动线程池

## 性能

`640×640` / `zidane.jpg` / Ryzen 5 5560U (6C/12T) / VS2022 `/O2 /arch:AVX2 /GL /LTCG`

| 模型 | 优化前 | 优化后 |
|------|--------|--------|
| yolov8n (80类) | 650 ms | **143 ms** |
| yolo11n | - | **163 ms** |
| yolo26n | - | **166 ms** |
| yolov10n | - | **184 ms** |
| yolov12n1 | - | **252 ms** |

> `SetInputSize(512)` 可再降约 36%。同机 ONNX Runtime 约 100ms，差距主要在 INT8 与更深度的打包微内核。

## 快速开始

### 易语言

```e
.版本 2
.DLL命令 InitModel, 整数型, "tiny_yolo_x86.dll", "InitModel", , , 模型路径, 文本型
.DLL命令 SetInputSize, , "tiny_yolo_x86.dll", "SetInputSize", , , 宽, 整数型, 高, 整数型
.DLL命令 YoloDetectFromMemory, 整数型, "tiny_yolo_x86.dll", "YoloDetectFromMemory", , , 图片字节集, 字节集, 图片大小, 整数型, 置信度, 小数型, NMS阈值, 小数型, 结果数组, 小数型, 数组, 最大数量, 整数型
.DLL命令 ReleaseModel, , "tiny_yolo_x86.dll", "ReleaseModel"

' 每组结果 6 个 float: [label, score, x, y, w, h]  x,y 为原图左上角坐标
' 内存加载（无需落盘，自动识别 onnx/tyro）:
' InitModelFromMemory(模型字节集, 长度)
```

### C++

```cpp
InitModel("yolov12n1.onnx");
SetInputSize(640, 640);
float results[100 * 6];
int n = YoloDetectFromMemory(imgData, imgSize, 0.25f, 0.45f, results, 100);
// results[i*6+0]=label, [1]=score, [2]=x, [3]=y, [4]=w, [5]=h
ReleaseModel();
```

### 命令行测试

```bat
cl /O2 /EHsc test_tiny.cpp /Fetest_tiny.exe
test_tiny.exe yolov12n1.onnx zidane.jpg 640
test_tiny.exe yolo11n.onnx zidane.jpg 640 0.25
```

## 编译

**环境:** Visual Studio 2022 + Windows SDK, CPU 需支持 AVX2/FMA

**x86 (易语言):**
```bat
:: x86 Native Tools Command Prompt
cl /c /O2 /GL /Gy /EHsc /MD /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo.obj
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X86 /LARGEADDRESSAWARE tiny_yolo.obj /OUT:tiny_yolo_x86.dll
```

**x64:**
```bat
:: x64 Native Tools Command Prompt
cl /c /O2 /GL /Gy /EHsc /MD /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo64.obj
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X64 tiny_yolo64.obj /OUT:tiny_yolo.dll
```

调试开关：`TINY_YOLO_PROFILE=1` 打印各算子耗时，`TINY_YOLO_THREADS=4` 强制线程数。

## API

| 导出函数 | 说明 |
|----------|------|
| `InitModel(path)` | 加载 `.onnx` 或 `.tyro`，返回 1 成功 |
| `InitModelFromMemory(data, size)` | 从内存加载，自动识别格式 |
| `SetInputSize(w,h)` | 设置输入分辨率，默认 640 |
| `YoloDetectFromMemory(img,size,conf,nms,results,max)` | 推理，返回检出数 |
| `ReleaseModel()` | 释放模型 |

## 目录

```
tiny_yolo.cpp         DLL 入口与后处理
tiny_yolo/
  model.h             图执行引擎 + 线程池
  ops.h               算子实现 (Conv/Winograd/GEMM/...)
  tensor.h            Tensor 与内存复用
  onnx_lite.h         极简 ONNX 解析
  onnx_convert.h      ONNX → 内部 OP + 融合
stb_image.h           图片解码
test_tiny.cpp         测试程序
```

## 支持的 YOLO 版本

| 版本 | 输出格式 | 备注 |
|------|----------|------|
| v8 / v11 / v12 | `[1, 4+nc, 8400]` | nc 任意 |
| v10 | `[1, 4+nc, 8400]` | 稀疏输出自动识别 |
| v26 | `[1, N, 6]` | NMS 已内置 |

## 致谢

- `stb_image` — Sean Barrett
- `ncnn` — 仅作参考对比
