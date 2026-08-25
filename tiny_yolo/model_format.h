// Tiny YOLO 推理引擎 - 自定义模型格式定义
// 二进制格式，避开 protobuf，最小化解析代码
#pragma once
#include <cstdint>

// 文件魔数 "TYO1" (Tiny YOLO v1)
#define TYO_MAGIC 0x314F5954

// 算子类型枚举
enum OpType : uint8_t {
    OP_Conv        = 0,
    OP_SiLU        = 1,   // x * sigmoid(x)，融合后的激活
    OP_Sigmoid     = 2,
    OP_Relu        = 3,
    OP_Add         = 4,
    OP_Mul         = 5,
    OP_Sub         = 6,
    OP_Div         = 7,
    OP_Exp         = 8,
    OP_Concat      = 9,
    OP_Resize      = 10,
    OP_Transpose   = 11,
    OP_Reshape     = 12,
    OP_MatMul      = 13,
    OP_Split       = 14,
    OP_Slice       = 15,
    OP_ReduceMax   = 16,
    OP_ReduceSum   = 17,
    OP_Softmax     = 18,
    OP_MaxPool     = 19,
    OP_Identity    = 20,
    OP_ConvSiLU    = 21,  // Conv + SiLU 融合
    OP_Unsqueeze   = 22,
    OP_Flatten     = 23,
    OP_Tile        = 24,
    OP_Gather      = 25,
    OP_GatherElements = 26,
    OP_TopK        = 27,
    OP_Mod         = 28,
    OP_Cast        = 29,
    OP_COUNT
};

// 文件头 (20 bytes)
#pragma pack(push, 1)
struct TyroHeader {
    uint32_t magic;          // TYO_MAGIC
    uint8_t  version;        // 格式版本 = 1
    uint8_t  input_count;    // 输入张量数量
    uint8_t  output_count;   // 输出张量数量
    uint16_t op_count;       // 算子数量
    uint16_t tensor_count;   // 张量总数（含常量和中间）
    uint32_t weights_size;   // 常量权重数据总大小 (bytes)
    uint32_t reserved;       // 保留
};

// 张量描述
struct TensorDesc {
    uint16_t id;             // 张量 ID
    uint8_t  ndim;           // 维度数
    uint8_t  dtype;          // 0=float32, 1=int64, 2=int32
    uint8_t  is_const;       // 是否常量权重
    uint8_t  reserved[3];
    uint32_t data_offset;    // 在权重数据区的偏移（is_const=1时有效）
    // 后面紧跟 ndim 个 int32 维度
};

// 算子描述
struct OpDesc {
    uint8_t  op_type;        // OpType
    uint8_t  input_count;    // 输入张量数量
    uint8_t  output_count;   // 输出张量数量
    uint8_t  attr_size;      // 属性数据大小 (bytes)
    // 后面紧跟：
    // 1. attr_size bytes 属性数据
    // 2. input_count 个 uint16 输入张量ID
    // 3. output_count 个 uint16 输出张量ID
};
#pragma pack(pop)

// Conv 属性 (固定大小)
#pragma pack(push, 1)
struct ConvAttr {
    int32_t kernel_h;
    int32_t kernel_w;
    int32_t stride_h;
    int32_t stride_w;
    int32_t pad_h;
    int32_t pad_w;
    int32_t dilation_h;
    int32_t dilation_w;
    int32_t group;
};
#pragma pack(pop)
