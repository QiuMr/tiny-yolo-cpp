// ONNX 到 TinyModel 内部格式的转换
#pragma once
#include "onnx_lite.h"
#include "model_format.h"
#include "tensor.h"
#include "ops.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>

// 属性类型（与convert_model.py一致）
enum AttrType : uint8_t {
    ATTR_FLOAT = 1,
    ATTR_INT = 2,
    ATTR_STRING = 3,
    ATTR_FLOATS = 4,
    ATTR_INTS = 5,
    ATTR_STRINGS = 6
};

// 写入属性到attr_data
static inline void write_attr_float(std::vector<uint8_t>& attr_data, const char* name, float v) {
    uint8_t nlen = (uint8_t)strlen(name);
    attr_data.push_back(nlen);
    for (uint8_t i = 0; i < nlen; i++) attr_data.push_back((uint8_t)name[i]);
    attr_data.push_back(ATTR_FLOAT);
    uint32_t fv;
    memcpy(&fv, &v, 4);
    // float存为4字节（与convert_model.py一致）
    attr_data.push_back((uint8_t)(fv & 0xFF));
    attr_data.push_back((uint8_t)((fv >> 8) & 0xFF));
    attr_data.push_back((uint8_t)((fv >> 16) & 0xFF));
    attr_data.push_back((uint8_t)((fv >> 24) & 0xFF));
}

static inline void write_attr_int(std::vector<uint8_t>& attr_data, const char* name, int64_t v) {
    uint8_t nlen = (uint8_t)strlen(name);
    attr_data.push_back(nlen);
    for (uint8_t i = 0; i < nlen; i++) attr_data.push_back((uint8_t)name[i]);
    attr_data.push_back(ATTR_INT);
    for (int i = 0; i < 8; i++) {
        attr_data.push_back((uint8_t)((v >> (i * 8)) & 0xFF));
    }
}

static inline void write_attr_string(std::vector<uint8_t>& attr_data, const char* name, const std::string& v) {
    uint8_t nlen = (uint8_t)strlen(name);
    attr_data.push_back(nlen);
    for (uint8_t i = 0; i < nlen; i++) attr_data.push_back((uint8_t)name[i]);
    attr_data.push_back(ATTR_STRING);
    uint16_t slen = (uint16_t)v.size();
    attr_data.push_back((uint8_t)(slen & 0xFF));
    attr_data.push_back((uint8_t)((slen >> 8) & 0xFF));
    for (uint16_t i = 0; i < slen; i++) attr_data.push_back((uint8_t)v[i]);
}

static inline void write_attr_ints(std::vector<uint8_t>& attr_data, const char* name, const std::vector<int64_t>& vals) {
    uint8_t nlen = (uint8_t)strlen(name);
    attr_data.push_back(nlen);
    for (uint8_t i = 0; i < nlen; i++) attr_data.push_back((uint8_t)name[i]);
    attr_data.push_back(ATTR_INTS);
    uint16_t cnt = (uint16_t)vals.size();
    attr_data.push_back((uint8_t)(cnt & 0xFF));
    attr_data.push_back((uint8_t)((cnt >> 8) & 0xFF));
    for (uint16_t i = 0; i < cnt; i++) {
        int64_t v = vals[i];
        for (int j = 0; j < 8; j++) {
            attr_data.push_back((uint8_t)((v >> (j * 8)) & 0xFF));
        }
    }
}

// ONNX op_type 到 OpType 的映射
static inline uint8_t map_op_type(const std::string& op_type) {
    if (op_type == "Conv") return OP_Conv;
    if (op_type == "ConvSiLU") return OP_ConvSiLU;
    if (op_type == "SiLU") return OP_SiLU;
    if (op_type == "Sigmoid") return OP_Sigmoid;
    if (op_type == "Relu") return OP_Relu;
    if (op_type == "Add") return OP_Add;
    if (op_type == "Mul") return OP_Mul;
    if (op_type == "Sub") return OP_Sub;
    if (op_type == "Div") return OP_Div;
    if (op_type == "Exp") return OP_Exp;
    if (op_type == "Concat") return OP_Concat;
    if (op_type == "Resize") return OP_Resize;
    if (op_type == "Transpose") return OP_Transpose;
    if (op_type == "Reshape") return OP_Reshape;
    if (op_type == "MatMul") return OP_MatMul;
    if (op_type == "Split") return OP_Split;
    if (op_type == "Slice") return OP_Slice;
    if (op_type == "ReduceMax") return OP_ReduceMax;
    if (op_type == "ReduceSum") return OP_ReduceSum;
    if (op_type == "Softmax") return OP_Softmax;
    if (op_type == "MaxPool") return OP_MaxPool;
    if (op_type == "Identity") return OP_Identity;
    if (op_type == "Unsqueeze") return OP_Unsqueeze;
    if (op_type == "Flatten") return OP_Flatten;
    if (op_type == "Tile") return OP_Tile;
    if (op_type == "Gather") return OP_Gather;
    if (op_type == "GatherElements") return OP_GatherElements;
    if (op_type == "TopK") return OP_TopK;
    if (op_type == "Mod") return OP_Mod;
    if (op_type == "Cast") return OP_Cast;
    return 0xFF; // 不支持
}

// 转换ONNX属性到.tyro属性格式
static inline void convert_attrs(const onnx_lite::OnnxNode& node, std::vector<uint8_t>& attr_data) {
    for (const auto& attr : node.attributes) {
        if (attr.type == 1) { // FLOAT
            if (!attr.floats.empty()) write_attr_float(attr_data, attr.name.c_str(), attr.floats[0]);
        } else if (attr.type == 2) { // INT
            if (!attr.ints.empty()) write_attr_int(attr_data, attr.name.c_str(), attr.ints[0]);
        } else if (attr.type == 3) { // STRING
            write_attr_string(attr_data, attr.name.c_str(), attr.s);
        } else if (attr.type == 6) { // FLOATS
            // 暂时只存第一个float，或者存为ints格式？
            // 实际上.tyro格式没有FLOATS类型，我们需要扩展
            // 暂时跳过
        } else if (attr.type == 7) { // INTS
            write_attr_ints(attr_data, attr.name.c_str(), attr.ints);
        } else if (attr.type == 8) { // STRINGS
            // 暂时跳过
        }
    }
}

// 图优化：识别Conv -> Sigmoid -> Mul (SiLU) 模式，融合为ConvSiLU
// 注意：ONNX中的SiLU通常是 x * sigmoid(x)，即Mul(x, Sigmoid(x))
static inline int fuse_conv_silu(std::vector<onnx_lite::OnnxNode>& nodes) {
    int fused = 0;
    std::unordered_map<std::string, size_t> output_to_node;
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out : nodes[i].outputs) {
            output_to_node[out] = i;
        }
    }

    std::vector<bool> removed(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); i++) {
        if (removed[i]) continue;
        if (nodes[i].op_type != "Conv") continue;

        // 找Conv的输出是否被Sigmoid使用
        std::string conv_out = nodes[i].outputs[0];
        int sigmoid_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "Sigmoid") continue;
            if (nodes[j].inputs[0] == conv_out) {
                sigmoid_idx = (int)j;
                break;
            }
        }
        if (sigmoid_idx < 0) continue;

        // 找Sigmoid的输出是否被Mul使用，且Mul的另一个输入是Conv的输出
        std::string sigmoid_out = nodes[sigmoid_idx].outputs[0];
        int mul_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "Mul") continue;
            if (nodes[j].inputs.size() != 2) continue;
            bool has_sigmoid = (nodes[j].inputs[0] == sigmoid_out || nodes[j].inputs[1] == sigmoid_out);
            bool has_conv = (nodes[j].inputs[0] == conv_out || nodes[j].inputs[1] == conv_out);
            if (has_sigmoid && has_conv) {
                mul_idx = (int)j;
                break;
            }
        }
        if (mul_idx < 0) continue;

        // use-count guard: conv_out must be consumed exactly twice
        // (Sigmoid + Mul) and sigmoid_out exactly once (Mul). Otherwise the Conv
        // output feeds another branch (residual/Concat) and renaming it breaks.
        {
            int conv_uses = 0, sig_uses = 0;
            for (size_t j = 0; j < nodes.size(); j++) {
                if (removed[j]) continue;
                for (const auto& in : nodes[j].inputs) {
                    if (in == conv_out) conv_uses++;
                    if (in == sigmoid_out) sig_uses++;
                }
            }
            if (conv_uses != 2 || sig_uses != 1) continue;
        }

        // 融合：把Conv改成ConvSiLU，输出改成Mul的输出
        nodes[i].op_type = "ConvSiLU";
        nodes[i].outputs[0] = nodes[mul_idx].outputs[0];
        removed[sigmoid_idx] = true;
        removed[mul_idx] = true;
        fused++;
    }

    // 移除被融合的节点
    std::vector<onnx_lite::OnnxNode> new_nodes;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!removed[i]) new_nodes.push_back(std::move(nodes[i]));
    }
    nodes = std::move(new_nodes);
    return fused;
}

// 图优化：识别ReduceMax -> Sub -> Exp -> ReduceSum -> Div 模式，融合为Softmax
static inline int fuse_softmax(std::vector<onnx_lite::OnnxNode>& nodes) {
    int fused = 0;
    std::unordered_map<std::string, size_t> output_to_node;
    for (size_t i = 0; i < nodes.size(); i++) {
        for (const auto& out : nodes[i].outputs) {
            output_to_node[out] = i;
        }
    }

    std::vector<bool> removed(nodes.size(), false);
    for (size_t i = 0; i < nodes.size(); i++) {
        if (removed[i]) continue;
        if (nodes[i].op_type != "ReduceMax") continue;

        // 找ReduceMax的输出是否被Sub使用
        std::string rm_out = nodes[i].outputs[0];
        int sub_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "Sub") continue;
            if (nodes[j].inputs[0] == rm_out || nodes[j].inputs[1] == rm_out) {
                sub_idx = (int)j;
                break;
            }
        }
        if (sub_idx < 0) continue;

        // 找Sub的输出是否被Exp使用
        std::string sub_out = nodes[sub_idx].outputs[0];
        int exp_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "Exp") continue;
            if (nodes[j].inputs[0] == sub_out) {
                exp_idx = (int)j;
                break;
            }
        }
        if (exp_idx < 0) continue;

        // 找Exp的输出是否被ReduceSum使用
        std::string exp_out = nodes[exp_idx].outputs[0];
        int rs_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "ReduceSum") continue;
            if (nodes[j].inputs[0] == exp_out) {
                rs_idx = (int)j;
                break;
            }
        }
        if (rs_idx < 0) continue;

        // 找ReduceSum的输出是否被Div使用
        std::string rs_out = nodes[rs_idx].outputs[0];
        int div_idx = -1;
        for (size_t j = 0; j < nodes.size(); j++) {
            if (removed[j]) continue;
            if (nodes[j].op_type != "Div") continue;
            if (nodes[j].inputs[0] == rs_out || nodes[j].inputs[1] == rs_out) {
                div_idx = (int)j;
                break;
            }
        }
        if (div_idx < 0) continue;

        // use-count guard: all four intermediates must have exactly one
        // consumer, else deleting these nodes leaves a branch with no producer.
        {
            auto uses_of = [&](const std::string& nm) {
                int n = 0;
                for (size_t j = 0; j < nodes.size(); j++) {
                    if (removed[j]) continue;
                    for (const auto& in : nodes[j].inputs) if (in == nm) n++;
                }
                return n;
            };
            if (uses_of(rm_out) != 1 || uses_of(sub_out) != 1
                || uses_of(exp_out) != 1 || uses_of(rs_out) != 1) continue;
        }

        // 融合：把ReduceMax改成Softmax，输出改成Div的输出
        // Softmax的输入是Sub的第一个输入（原始输入）
        std::string softmax_input = nodes[sub_idx].inputs[0];
        if (softmax_input == rm_out) softmax_input = nodes[sub_idx].inputs[1];
        nodes[i].op_type = "Softmax";
        // The original ReduceMax carries axes/keepdims, but op_softmax looks
        // for "axis" and breaks on the first other attribute - set it here.
        nodes[i].attributes.clear();
        {
            onnx_lite::OnnxNode::Attribute ax;
            ax.name = "axis";
            ax.type = 2; // INT
            ax.ints = { -1 };
            nodes[i].attributes.push_back(std::move(ax));
        }
        nodes[i].inputs[0] = softmax_input;
        nodes[i].outputs[0] = nodes[div_idx].outputs[0];
        removed[sub_idx] = true;
        removed[exp_idx] = true;
        removed[rs_idx] = true;
        removed[div_idx] = true;
        fused++;
    }

    // 移除被融合的节点
    std::vector<onnx_lite::OnnxNode> new_nodes;
    for (size_t i = 0; i < nodes.size(); i++) {
        if (!removed[i]) new_nodes.push_back(std::move(nodes[i]));
    }
    nodes = std::move(new_nodes);
    return fused;
}
