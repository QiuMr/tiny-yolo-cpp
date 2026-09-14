# Tiny YOLO

手写 C++ YOLO 推理引擎，零第三方依赖，AVX2 加速，编译为 DLL 供易语言 / C++ 直接调用。

> 纯 C++ 实现全部算子，无 OpenCV / ONNX Runtime 依赖。ONNX 直载，兼容 YOLOv8 / v10 / v11 / v12 / v26 与自定义训练模型，附开箱即用的易语言模块。

![C++17](https://img.shields.io/badge/C++-17-00599C) ![Win32](https://img.shields.io/badge/Win32-x86%20%7C%20x64-555) ![License](https://img.shields.io/badge/license-MIT-green) ![DLL](https://img.shields.io/badge/DLL-214KB%20%7C%20256KB-brightgreen)

---

## 特性

- **零依赖** - 仅 stb_image + Win32 API；静态 CRT + UPX 后 x86 约 214KB
- **ONNX 直载** - InitModel("yolo11n.onnx") 即可，自动融合 Conv+SiLU、Softmax
- **内存加载** - InitModelFromMemory(data, size)，支持模型嵌入资源表 / 网络下发
- **易语言友好** - __stdcall 导出 + 现成易语言模块，三行代码完成识别
- **全系支持** - v8 / v10 / v11 / v12 / v26 自动识别，任意类别数
- **高性能** - AVX2+FMA、Winograd F(2,3)/F(4,3)（F4 变换 AVX2 批量 SIMD）、W 打包 GEMM、事件驱动线程池

## 性能

640x640 / zidane.jpg / Ryzen 5 5560U (6C/12T) / VS2022 /O2 /arch:AVX2

| 模型 | 初版 | 当前 |
|------|------|------|
| yolov8n (80类) | 650 ms | **~110 ms** |
| yolo26n | 166 ms | **~130 ms** |
| yolo11n | 163 ms | **~130 ms** |
| yolov10n | 184 ms | **~150 ms** |
| yolov12n1 (自定义) | 252 ms | **~205 ms** |
| Cr-黄瓦v11m256 (256x256, 自定义) | ~111 ms | **~110 ms (冷机, SIMD)** |

> v11m256 冷机对标（同一热量窗口 A/B）：Winograd 标量 128ms → F4 SIMD 110ms（-14%）。
> 5560U 为 15W U 系列，持续负载受功耗墙限制（实测线程数 4~12 耗时几乎相同）。
> 同机 ONNX Runtime 256 输入约 44ms，差距主要在 INT8 与更深度的打包微内核。
> 2026-09-08 一轮 Bug 修复 + 预处理优化后实测（yolo11n@640 / 1.jpg / x86 Release）：
> **143ms → 136ms**，检出框与修复前逐位一致（无精度退化）。

## 快速开始

### 易语言（推荐：导入现成模块）

1. 易语言新建「易模块」，粘贴《易语言_YOLO识别模块.txt》内容，保存为 .ec（仓库也提供编译好的 YOLO推理引擎.ec）
2. 工具 → 易模块管理器 → 导入并勾选使用
3. 把 tiny_yolo_x86.dll 和模型文件放到 exe 同目录

```e
.如果真 (加载模型 ("yolo11n.onnx"))
    识别文件 ("zidane.jpg", , , 结果集)          ' 结果集为"检测结果"数组
    .计次循环首 (取数组成员数 (结果集), i)
        调试输出 (结果集 [i].类别, 结果集 [i].置信度,
                  结果集 [i].左边, 结果集 [i].顶边,
                  结果集 [i].宽度, 结果集 [i].高度)
    .计次循环尾 ()
    释放模型 ()
.如果真结束
```

完整示例见《易语言_使用示例.txt》。常用姿势：

```e
加载模型 ("yolo11n.onnx", 512)              ' 512 分辨率换约 30% 提速（固定尺寸模型必须用其训练分辨率）
加载模型 ("Cr-黄瓦v11m256.onnx")            ' 省略尺寸：自动采用模型声明的 256x256
加载模型数据 (模型字节集)                    ' 模型不落盘：资源表 / 网络下发
识别图片数据 (截图字节集, , , 结果集)         ' 直接识别字节集（截屏/网络图）
调试输出 (取错误信息 ())                     ' 失败原因
```

### 易语言（原始 DLL 声明）

```e
.版本 2

.DLL命令 InitModel, 整数型, "tiny_yolo_x86.dll", "_InitModel@4"
    .参数 model_path, 文本型

.DLL命令 YoloDetectFromMemory, 整数型, "tiny_yolo_x86.dll", "_YoloDetectFromMemory@24"
    .参数 img_data, 字节集
    .参数 img_size, 整数型
    .参数 conf_thres, 小数型
    .参数 nms_thres, 小数型
    .参数 results, 小数型, 数组
    .参数 max_size, 整数型

.DLL命令 ReleaseModel, , "tiny_yolo_x86.dll", "_ReleaseModel@0"

' results 每个目标占 6 个 float: [label, score, x, y, w, h]，x,y 为原图左上角坐标
' 内存加载（无需落盘，自动识别 onnx/tyro）:
' InitModelFromMemory(模型字节集, 长度)
```

### C++

```cpp
InitModel("yolo11n.onnx");
SetInputSize(640, 640);
float results[100 * 6];
int n = YoloDetectFromMemory(imgData, imgSize, 0.25f, 0.45f, results, 100);
// results[i*6+0]=label, [1]=score, [2]=x, [3]=y, [4]=w, [5]=h
ReleaseModel();
```

### 命令行测试

```bat
cl /O2 /EHsc test_tiny.cpp /Fetest_tiny.exe
test_tiny.exe yolo11n.onnx zidane.jpg 640 0.25
```

### 回归测试

`test_fixes.cpp` 覆盖**已经修掉的历史缺陷**，每个用例在修复前都会失败，用于防止回归：

```bat
:: x86 Native Tools Command Prompt，需先按"编译"一节生成 tiny_yolo_x86.dll
cl /O2 /EHsc test_fixes.cpp /Fetest_fixes.exe
test_fixes.exe          :: 同目录需有 1.jpg / tall.jpg 与用到的 .onnx
```

| 用例 | 覆盖的缺陷 |
|------|-----------|
| T1 | `ReleaseModel` 后换模型，被上一次 `SetInputSize` 残留的尺寸静默拦截（永久 0 检出） |
| T2 | `InitModelFromMemory` 曾把模型写到 `%TEMP%` 且从不删除 |
| T3 | 未释放时重复 `InitModel` 静默沿用旧模型 |
| T4 | letterbox 缓冲跨帧复用，上一帧像素污染填充区（曾造成误检） |
| T5 | 置信度 / NMS 阈值为 0 时缺少兜底 |
| T6 | v8 / v10 / v26 多模型交叉冒烟 |

> 修复前 **13 通过 / 4 失败**，修复后 **17 通过 / 0 失败**。

## API

| 导出函数 | 说明 |
|----------|------|
| InitModel(path) | 加载 .onnx 或 .tyro 文件，返回 1 成功 |
| InitModelFromMemory(data, size) | 从内存加载，自动识别格式。**不写任何临时文件**，返回后调用方可立即释放字节集 |
| SetInputSize(w,h) | 设置输入分辨率，默认 640；未显式调用时自动采用模型声明的输入尺寸 |
| YoloDetectFromMemory(img,size,conf,nms,results,max) | 推理，返回检出数 |
| ReleaseModel() | 释放模型 |

## 编译

**环境:** Visual Studio 2022 + Windows SDK，CPU 需支持 AVX2/FMA

产物为**静态 CRT (/MT) 零依赖 + UPX 压缩**，目标机器无需安装任何运行库。
一键脚本：build_x86.cmd（易语言用）/ build_x64.cmd，或手动：

**x86 (易语言):**
```bat
:: x86 Native Tools Command Prompt
cl /c /O2 /GL /Gy /EHsc /MT /DNDEBUG /D_CRT_SECURE_NO_WARNINGS /arch:AVX2 tiny_yolo.cpp /Fotiny_yolo.obj
link /DLL /LTCG /OPT:REF /OPT:ICF /MACHINE:X86 /LARGEADDRESSAWARE tiny_yolo.obj /OUT:tiny_yolo_x86.dll
upx -9 tiny_yolo_x86.dll
```

**x64:** 同上，换 vcvars64.bat、/MACHINE:X64，输出 tiny_yolo.dll。
体积参考（UPX -9 后）：x86 约 214KB，x64 约 256KB（UPX -9 --lzma 可再降至 198/228KB）。

## 模型转换

.onnx 可直接加载；如需更小的自有格式，用 convert_model.py 转换：

```bash
pip install onnx numpy
python convert_model.py yolo11n.onnx yolo11n.tyro
```

## 支持的 YOLO 版本

| 版本 | 输出格式 | 备注 |
|------|----------|------|
| v8 / v11 / v12 | [1, 4+nc, 8400] | nc 任意，自定义训练模型可用 |
| v10 | [1, 4+nc, 8400] | 稀疏输出自动识别（稀疏率+几何双重判定） |
| v26 | [1, N, 6] | NMS 已内置 |

## 调试开关

| 环境变量 | 作用 |
|----------|------|
| TINY_YOLO_PROFILE=1 | 打印各算子耗时 |
| TINY_YOLO_NAN_DEBUG=1 | 逐算子扫描 NaN/Inf 并定位源头 |
| TINY_YOLO_DUMP=1 | 打印输出张量布局、letterbox 参数与最优 anchor |
| TINY_YOLO_NO_SILU=1 | 禁用融合 SiLU（定位数值问题） |
| TINY_YOLO_THREADS=N | 强制线程数 |
| TINY_YOLO_NO_SIMD=1 | 关闭 Winograd F4 SIMD 变换（数值对照/定位性能用） |
| TINY_YOLO_LOG=1 | 打印模型加载期日志（节点数 / 融合数 / 权重大小）。**默认关闭**：易语言等 GUI 宿主没有控制台 |

辅助工具源码：ort_dump.cpp（ONNX Runtime 对照基准）、test_weights.cpp（权重 NaN 扫描）、
test_s2.cpp / test_wino.cpp（卷积内核对拍测试）、test_initmem.cpp（内存加载路径测试）。

## 开发笔记

<details>
<summary><b>性能优化记录</b></summary>

- Winograd 权重变换缓存命中后直接引用，去掉每帧整块 memcpy
- 1x1 卷积 W 打包缓存（[K][N/6][6] 布局提升 FMA 缓存命中）
- 预处理输入张量跨帧复用（letterbox 边界只在首帧填充）
- Winograd F(4,3) 输入/输出变换 AVX2 批量 SIMD：8 tile/车道 gather/loadu，与标量逐位全等（test_simd.cpp，maxdiff=0），冷机 -18ms
- 后处理单次扫描完成 V10 判定与框解析（原来扫两遍）
- Winograd F(4,3) 支持（C_in*C_out ≥ 2048 自动启用，算力富余的 CPU 受益）
- 修复并保留 3x3 s2 直接卷积内核（原实现有两处数学错误），实测该机 im2col+GEMM 更快故默认关闭

</details>

<details>
<summary><b>已修复的数值/后处理 Bug</b></summary>

- fast_exp（Schraudolph 近似）NaN 毒区：12102203*x + 1064866805 在 x < -87.9895 时整数结果为负，reinterpret 成 float 即 NaN。原 ±88 钳制放过了 (-88, -87.99) 区间，大激活值经 SiLU 会产生 NaN 并向全图扩散（表现为检出框全是 NaN、置信度全为 1）。现已全线钳制到 ±86（标量守卫 + 所有 AVX 路径），对精度无影响。
- V10 自动判定误触发：仅凭"活跃 anchor < 100"会把类别少、背景抑制强的自定义模型误判成 v10 端到端格式，把 (cx,cy,w,h) 按 (x1,y1,x2,y2) 解读，输出负宽高。现改为稀疏率（精确零分数 >90%）+ 几何交叉验证（xyxy 需满足 x2>x1 且 y2>y1）双重判定。
- V10 判定假阴性：yolov10n 等端到端导出的 cls 背景分值不是精确 0 而是 1e-4~1e-5 的小正数，"精确零分数"恒为 0，导致 v10 永远走 v8 解析路径、坐标错乱。判定改为"峰值分 ≤1e-4 的 anchor 占比 >90%"，v10n 实测与 ONNX Runtime 逐框一致。
- 固定尺寸模型输入不匹配：v11m256 等按固定分辨率导出的模型（DFL/注意力 Reshape 硬编码），若调用方未传输入尺寸，引擎默认 640 推理会产出垃圾框（conf=1.0、数量随机）。现在 InitModel/InitModelFromMemory 成功后会读取模型声明的输入尺寸并自动采用；显式调用 SetInputSize 后以显式值为准。
- 框坐标偏移：v8/v10/v11 后处理把中心点 (cx,cy) 当左上角输出，所有框相对正确位置偏移半个框的大小。现已改为左上角 = 中心 - 半宽高，坐标与 ONNX Runtime（test.py）逐框一致。
- 性能：新增专用 depthwise 3x3 卷积内核（C2PSA 注意力深度卷积此前走通用分组路径，逐通道串行 im2col+GEMM）；Winograd 输入变换在小分辨率下改用密集暂存 + 连续 k 平面写入。256 输入自定义模型约 111ms → ~91ms（本机）。

</details>

<details>
<summary><b>已修复的稳定性 / 接口 Bug（2026-09-08）</b></summary>

- **`ReleaseModel` 不重置输入尺寸标记（最坑）**：`SetInputSize(512)` → `释放模型()` → `加载模型(另一个模型)`，第二次加载仍沿用 512，与模型声明尺寸不符时被尺寸守卫**静默拦截，永远返回 0 个检出且无任何报错**。现在释放模型与输入尺寸状态一起重置。
- **letterbox 缓冲跨帧复用污染**：输入张量缓冲只在分辨率变化时才填 114 灰，之后每帧只覆写图像窗口。宽高比变化后，上一帧写过而本帧覆盖不到的区域**保留旧像素**，被模型当成真实内容——实测在一张 320x800 图上凭空多出 3 个误检框。现在几何一旦变化就整块重填（同尺寸连续帧仍走快路径）。
- **`InitModelFromMemory` 名不副实**：实际把模型写到 `%TEMP%\tiny_yolo_model.onnx`（固定名、从不删除、多进程冲突），既落盘又泄漏。现在从调用方缓冲直接解析；两个加载器都会把权重拷进自有内存，调用返回后即可释放字节集。
- **临时缓冲分配失败未判空**：`get_temp_buf` 失败返回 `nullptr`，6 个调用点全部直接使用（Winograd 大层单次申请可达 ~70MB，32 位下失败即崩溃）。现在改为抛 `std::bad_alloc`，由 `run()` 统一捕获转成错误码。
- **`get_output` 靠猜**：直接取 `ops.back().output_ids[idx]`，且从不检查越界（`name_to_id` 成员声明后从未填充）。现在按 `output_names` 解析真实输出 id，并对下标与张量 id 做边界检查。
- **重复 `InitModel` 静默沿用旧模型**：`if (g_model) return 1;`，换模型无效还返回成功。现在会先释放旧模型。
- **阈值缺少兜底**：`conf_thres=0` 会让全部 8400 个 anchor 进入 O(n²) NMS（表现为卡死）。现在 0 / 负数 / NaN 一律回落到 0.25 / 0.45。
- **其他**：`Tensor::alloc` 与图片缩放缓冲的 `malloc` 不判空；张量元素数用 `int` 连乘存在溢出风险（已加 `size_t` 检查）；加载期无条件 `fprintf(stderr)`（GUI 宿主无控制台，已改为 `TINY_YOLO_LOG=1` 才输出）；`InitModelFromMemory` 对未对齐缓冲做 `uint32_t` 强转（改为逐字节比较）。

</details>

<details>
<summary><b>性能优化记录（2026-09-08）</b></summary>

- 预处理归一化改用 256 项查找表替代 `x / 255.0f`：640 输入每帧省掉约 120 万次除法
- `run()` 里每算子无条件调用 `chrono::now()`（一次 QueryPerformanceCounter），改为仅在开启 `TINY_YOLO_PROFILE` 时计时
- 后处理的 `best_score` / `best_cls` 改为 `thread_local` 复用，不再每帧申请约 67KB
- 合计：yolo11n@640 由 143ms 降至 136ms，检出框逐位一致

</details>

<details>
<summary><b>已修复的数值/正确性 Bug（2026-09-14）</b></summary>

- **Constant 节点产出的 int64 张量被降级成 float**：ONNX `Constant` 分支无条件把 dtype 写成 FLOAT（int64 值转 float 存 4 字节），而 Reshape / Slice / ReduceMax 的消费端在 `dtype != INT64/INT32` 时按 **8 字节**读同一块内存，shape 解析出垃圾值后静默出错图。现在保留原始 dtype，int64 按 8 字节存储。
- **Reshape / Slice / ReduceMax 的兜底分支按 int64 错读**：这三处的 `else` 落到 `(const int64_t*)data`，与 Unsqueeze / Tile / TopK 已有的「按 float 读」约定相反。已统一为按 4 字节 float 读。
- **`get_output` 查名失败回落到输入张量**：`output_tensor_ids` 用 `0` 作未解析标记，而 id 0 恰好是第一个输入张量，`get_output` 会返回错张量当检测输出。改用 `0xFFFF` 哨兵并返回 `nullptr`。
- **Conv+SiLU / Softmax 融合不查消费者数**：Conv 输出若还被残差 / Concat 分支消费，融合后改名会让那些节点读到无生产者的张量。现在融合前要求 conv_out 恰好被 2 个节点消费、sigmoid_out 恰好 1 个（Softmax 同理要求 4 个中间张量各只有唯一消费者）。
- **融合出的 Softmax 属性残留**：原 ReduceMax 带的是 `axes`/`keepdims`，而 `op_softmax` 找的是 `axis` 且遇到第一个非 axis 属性就 break —— 之前靠「DFL 的 softmax 正好在末维」这个巧合工作。现在显式清属性并写入 `axis=-1`。
- **分组卷积缓冲分配的恒真条件**：`max_numel = max(a,b)` 后再判 `a+b > max(a,b)` 永远成立，那条「复用 temp buffer 后半段」的分支是死代码；一旦生效，收尾的 `free()` 会释放 temp buffer 内部指针（堆破坏）。已改为独立 `malloc` + 判空抛 `bad_alloc`。
- **`SetInputSize` 先于 `InitModel` 时尺寸被静默清掉**：`InitModel` 开头的 `release_model()` 会重置显式尺寸标记。现在内部调用保留用户刚设的尺寸，仅用户主动 `ReleaseModel()` 时清除（T1 回归不受影响）。

</details>

<details>
<summary><b>性能改动（2026-09-14）</b></summary>

- `gemm_nm_core` 的 N 分块 `NB` 6 → 8，标量尾循环同步扩到 8 路，AVX2 累加器从 6 个增到 8 个
- **`NB` 必须与 `gemm_nm()` 的 `CH` 同值**：实测 `NB=6` 配 `CH=8` 会把 8 通道块拆成 6+2，第二次调用 `ni=2` 严重欠用寄存器，比 `CH=6` 还慢 2~6ms
- 非 packed 路径 `CH` 6 → 8；`mb` 上限 8 → 32（12 线程不再有空转）
- 线程数改为 `hardware_concurrency() - 1`（主线程也参与 `parallel_for`，避免超订）

> 注：本轮性能改动**未取得可信的 A/B 结论**。同机对照中同一个基线 DLL 在不同轮次测出 117ms 与 128ms（约 9% 漂移），大于预期收益；`NB`/`CH` 同值这条是唯一在两轮对照中方向一致的结论。检出框与修复前逐位一致，无精度退化。

</details>

## 目录

```
tiny_yolo.cpp         DLL 入口与后处理
tiny_yolo/
  ops.h               算子实现 (Conv/Winograd F4 SIMD/GEMM/...)
test_simd.cpp         Winograd F4 SIMD 变换与标量对拍单测 (maxdiff=0)
tiny_yolo/
  model.h             图执行引擎 + 线程池
  ops.h               算子实现 (Conv/Winograd/GEMM/...)
  tensor.h            Tensor 与内存复用
  onnx_lite.h         极简 ONNX 解析
  onnx_convert.h      ONNX → 内部 OP + 融合
convert_model.py      ONNX → .tyro 转换器
stb_image.h           图片解码
test_tiny.cpp         测试程序
test_fixes.cpp        Bug 回归测试（17 项，见"回归测试"一节）
tall.jpg              回归测试 T4 用的极端宽高比图（320x800）
易语言_YOLO识别模块.txt  易语言模块源码(导入即用)
易语言_使用示例.txt      易语言三行用法示例
YOLO推理引擎.ec        编译好的易语言模块
tiny_yolo_x86.dll     预编译 x86 DLL (/MT+UPX, 213KB)
tiny_yolo.dll         预编译 x64 DLL (/MT+UPX, 251KB)
```

## 致谢

- stb_image — Sean Barrett
- ncnn — 仅作参考对比

## 许可证

MIT
