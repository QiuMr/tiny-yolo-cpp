# Tiny YOLO — 手写 C++ YOLO 推理引擎 (32位 DLL / 易语言)

> 纯 C++ 手写算子、无第三方依赖、AVX2 加速、直接加载 ONNX，一键编译为 32 位 DLL 供易语言调用。支持 YOLOv8 / v10 / v11 / v12 / v26 全系。

![C++](https://img.shields.io/badge/C++-17-blue) ![Platform](https://img.shields.io/badge/Platform-Win32%20%7C%20x64-lightgrey) ![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ 特性

- **零依赖**：仅 `stb_image` + Windows API，不依赖 OpenCV / ONNX Runtime / ncnn
- **ONNX 直载**：`InitModel("yolo11n.onnx")` 即可，内部自动解析 + `Conv+SiLU` / `Softmax` 融合
- **兼容旧格式**：同时支持自研 `.tyro` 二进制格式（自动识别）
- **易语言友好**：`__stdcall` 导出，32 位 DLL，开箱即用
- **全系 YOLO**：v8 / v10(end-to-end) / v11 / v12(含注意力) / v26(NMS内置) 自动识别
- **AVX2 + 多线程**：Winograd F(2,3) / 1x1 直通 GEMM / 事件驱动线程池

---

## 📊 性能

测试环境：AMD Ryzen 5 5560U (6C/12T) / 640×640 / `zidane.jpg` / Release + `/arch:AVX2` + `/O2 /GL /LTCG`

| 模型 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| yolov8n.onnx (COCO 80类) | ~650 ms | **~150 ms** | **4.3×** |
| yolo11n.onnx | - | **~165 ms** | - |
| yolo26n.onnx | - | **~170 ms** | - |
| yolov10n.onnx | - | **~185 ms** | - |
| yolov12n1.onnx (注意力版) | - | **~255 ms** | 2.5× |

> 单层 Winograd 卷积扩展性：1 线程 3.47ms → 8 线程 0.95ms (2.2×)。ONNX Runtime 同机约 100ms，差距主要来自 INT8/更深度的打包微内核。

---

## 📁 目录结构

```
.
├── tiny_yolo.cpp          # DLL 入口 & C 接口 (InitModel / YoloDetectFromMemory / ReleaseModel)
├── tiny_yolo/
│   ├── model.h            # 图执行引擎 + 线程池
│   ├── ops.h              # 全部算子实现 (Conv/Winograd/GEMM/Transpose/Softmax...)
│   ├── tensor.h           # Tensor & 内存复用
│   ├── model_format.h     # .tyro 格式定义
│   ├── onnx_lite.h        # 极简 ONNX protobuf 解析
│   └── onnx_convert.h     # ONNX → 内部 OP 转换 + 融合
├── stb_image.h / stb_image_resize2.h
├── test_tiny.cpp          # 命令行测试程序
├── yolo11n.onnx / yolov8n.onnx / ...  # 示例模型 (建议用 Git LFS / Release 托管)
├── ncnn-master/           # 参考对比 (仅源码)
└── README.md
```

---

## 🔨 编译

### 环境
- Visual Studio 2022 (v143) + Windows SDK 10
- 需 CPU 支持 AVX2 + FMA（近 10 年主流 CPU 均支持）

### x86 (易语言用)

```bat
:: x86 Native Tools Command Prompt
cl /c /O2 /GL /Gy /EHsc /MD /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo.obj
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X86 /LARGEADDRESSAWARE tiny_yolo.obj /OUT:tiny_yolo_x86.dll
:: 如需兼容旧测试程序，同时复制一份
copy tiny_yolo_x86.dll tiny_yolo_x86_new.dll
```

### x64

```bat
:: x64 Native Tools Command Prompt
cl /c /O2 /GL /Gy /EHsc /MD /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo64.obj
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X64 tiny_yolo64.obj /OUT:tiny_yolo.dll
```

### 测试程序

```bat
cl /O2 /EHsc test_tiny.cpp /Fetest_tiny.exe
test_tiny.exe yolov12n1.onnx zidane.jpg 640
test_tiny.exe yolo11n.onnx zidane.jpg 640 0.25  # 可选: 第4参数为输入尺寸，第5参数为置信度阈值
```

> 调试开关：`set TINY_YOLO_PROFILE=1` 后运行会在 stderr 打印各算子耗时；`set TINY_YOLO_THREADS=4` 可强制线程数。

---

## 📖 易语言调用

```e
.版本 2

.DLL命令 InitModel, 整数型, "tiny_yolo_x86.dll", "InitModel"
    .参数 模型路径, 文本型
.DLL命令 SetInputSize, , "tiny_yolo_x86.dll", "SetInputSize"
    .参数 宽, 整数型
    .参数 高, 整数型
.DLL命令 YoloDetectFromMemory, 整数型, "tiny_yolo_x86.dll", "YoloDetectFromMemory"
    .参数 图片数据, 字节集
    .参数 图片大小, 整数型
    .参数 置信度阈值, 小数型
    .参数 NMS阈值, 小数型
    .参数 结果缓冲区, 小数型, 数组
    .参数 最大数量, 整数型
.DLL命令 ReleaseModel, , "tiny_yolo_x86.dll", "ReleaseModel"

' 结果缓冲区: 每6个float为一组 [label, score, x, y, w, h] (x,y为左上角)
```

**内存加载 ONNX**（从资源/网络下载的字节集直接加载，无需落盘）：

```e
InitModelFromMemory (模型字节集, 字节集长度)  ' 自动识别 tyro/onnx
```

---

## 🔧 C++ API

```cpp
int __stdcall InitModel(const char* path); // .onnx 或 .tyro
int __stdcall InitModelFromMemory(unsigned char* data, int size);
void __stdcall SetInputSize(int w, int h); // 默认 640
int __stdcall YoloDetectFromMemory(unsigned char* img, int img_size,
                                   float conf, float nms,
                                   float* results, int max_size);
void __stdcall ReleaseModel();
```

`results` 布局：`[label, conf, x, y, w, h] * N`，坐标已还原到原图。

---

## 🧠 支持的 YOLO 版本

| 版本 | 输出格式 | 备注 |
|------|----------|------|
| v8 / v11 / v12 | `[1, 4+nc, 8400]` | nc 任意，自动 NMS |
| v10 | `[1, 4+nc, 8400]` | 稀疏输出，启发式识别 |
| v26 | `[1, N, 6]` (x1,y1,x2,y2,conf,cls) | NMS 已内置 |

后处理已兼容任意类别数（不限 80 类）。

---

## 🚀 优化历程（本轮）

- 修复 32 位构建线程池被 `#ifdef _WIN64` 禁用 → 8 线程事件驱动池
- Winograd 权重缓存 + 去 V 转置 + 阶段并行化 + SiLU 融合
- 新写 `gemm_nm` 内核：`Out[N,M]=W[N,K]·X[K,M]` 零转置，二维分块
- 3×3 s2 改 `[K,M]` 布局 im2col，padding/im2col 并行化
- 属性/ dtype 双格式兼容（旧 tyro ↔ ONNX 原生码）
- 8×8 块转置（洗牌指令）、Softmax 向量化、Resize ×2 特化

650ms → 150ms (yolov8n) / 255ms (yolov12n1)

---

## 🔮 下一步

- **输入尺寸**：`SetInputSize(512)` 可再省 ~36% 计算量
- **INT8**：Zen2 无 VNNI，当前 fp32 已接近峰值；Zen4 / Intel 12代+ 可上 W8A8 预计直达 100ms 内
- **更深打包微内核**（ncnn 式 6×16 面板）

---

## 📄 许可

MIT

---

## 🙏 致谢

- `stb_image` / `stb_image_resize2` — Sean Barrett
- `ncnn` — 腾讯优图（仅作参考对比）
