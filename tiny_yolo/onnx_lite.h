// 杞婚噺绾NNX瑙ｆ瀽鍣紙鍙В鏋怸OLO鎺ㄧ悊闇€瑕佺殑瀛楁锛?
// 涓嶄緷璧杙rotobuf搴擄紝鎵嬪啓Protobuf wire format瑙ｆ瀽
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>

namespace onnx_lite {

// Protobuf wire types
enum WireType {
    WIRE_VARINT = 0,
    WIRE_FIXED64 = 1,
    WIRE_LENGTH_DELIMITED = 2,
    WIRE_FIXED32 = 5
};

// 璇诲彇varint
static inline uint64_t read_varint(const uint8_t*& p, const uint8_t* end) {
    uint64_t result = 0;
    int shift = 0;
    while (p < end) {
        uint8_t b = *p++;
        result |= (uint64_t)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift >= 64) throw std::runtime_error("varint too long");
    }
    return result;
}
static inline uint32_t read_tag(const uint8_t*& p, const uint8_t* end, uint32_t& wire_type) {
    uint64_t tag = read_varint(p, end);
    wire_type = (uint32_t)(tag & 0x07);
    return (uint32_t)(tag >> 3);
}

// 璺宠繃涓€涓瓧娈?
static inline void skip_field(const uint8_t*& p, const uint8_t* end, uint32_t wire_type) {
    switch (wire_type) {
        case WIRE_VARINT:
            read_varint(p, end);
            break;
        case WIRE_FIXED64:
            p += 8;
            break;
        case WIRE_LENGTH_DELIMITED: {
            uint64_t len = read_varint(p, end);
            p += len;
            break;
        }
        case WIRE_FIXED32:
            p += 4;
            break;
        default:
            // 瀹归敊锛氭湭鐭ire type锛屽皾璇曡烦杩?瀛楄妭
            if (p < end) p++;
            break;
    }
    if (p > end) p = end; // 瀹归敊
}

// ONNX TensorProto data types
enum TensorDataType {
    TENSOR_UNDEFINED = 0,
    TENSOR_FLOAT = 1,
    TENSOR_UINT8 = 2,
    TENSOR_INT8 = 3,
    TENSOR_UINT16 = 4,
    TENSOR_INT16 = 5,
    TENSOR_INT32 = 6,
    TENSOR_INT64 = 7,
    TENSOR_STRING = 8,
    TENSOR_BOOL = 9,
    TENSOR_FLOAT16 = 10,
    TENSOR_DOUBLE = 11
};

// 寮犻噺
struct OnnxTensor {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = TENSOR_UNDEFINED;
    std::vector<float> float_data;  // 解析后的float数据
    std::vector<int64_t> int64_data;  // 解析后的int64数据
    // raw_data存储原始字节
    std::vector<uint8_t> raw_data;
};

// 鑺傜偣锛堢畻瀛愶級
struct OnnxNode {
    std::string op_type;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string name;
    // 灞炴€э紙鍙瓨鍘熷瀛楄妭锛岄渶瑕佹椂鍐嶈В鏋愶級
    struct Attribute {
        std::string name;
        int32_t type = 0;
        std::vector<int64_t> ints;
        std::vector<float> floats;
        std::string s;
        OnnxTensor t;
        bool has_t = false;
    };
    std::vector<Attribute> attributes;
};

// 鍊间俊鎭紙杈撳叆杈撳嚭锛?
struct OnnxValueInfo {
    std::string name;
    std::vector<int64_t> dims;
    int32_t data_type = TENSOR_UNDEFINED;
};

// ONNX妯″瀷
struct OnnxModel {
    std::vector<OnnxNode> nodes;
    std::vector<OnnxTensor> initializers;  // 鏉冮噸
    std::vector<OnnxValueInfo> inputs;
    std::vector<OnnxValueInfo> outputs;
    std::vector<OnnxValueInfo> value_info;  // 涓棿寮犻噺

    // 鎸夊悕瀛楁煡鎵炬潈閲?
    const OnnxTensor* find_initializer(const std::string& name) const {
        for (const auto& t : initializers) {
            if (t.name == name) return &t;
        }
        return nullptr;
    }
};

// 瑙ｆ瀽TensorProto
static inline void parse_tensor_proto(const uint8_t* p, const uint8_t* end, OnnxTensor& tensor) {
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        switch (field_num) {
            case 1: { // dims (repeated int64)
                if (wire_type == 0) { // non-packed
                    int64_t dim = (int64_t)read_varint(p, end);
                    if (dim > 0 && dim < 100000) {
                        tensor.dims.push_back(dim);
                    }
                } else if (wire_type == 2) { // packed
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    const uint8_t* sub_end = p + len;
                    while (p < sub_end) {
                        int64_t dim = (int64_t)read_varint(p, sub_end);
                        if (dim > 0 && dim < 100000) {
                            tensor.dims.push_back(dim);
                        }
                    }
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 2: { // data_type (int32)
                if (wire_type == 0) {
                    tensor.data_type = (int32_t)read_varint(p, end);
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 4: { // float_data (repeated float, packed or non-packed)
                if (wire_type == 2) { // packed
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    size_t count = len / 4;
                    for (size_t i = 0; i < count; i++) {
                        float f;
                        memcpy(&f, p + i * 4, 4);
                        tensor.float_data.push_back(f);
                    }
                    p += len;
                } else if (wire_type == 5) { // non-packed, single float
                    float f;
                    memcpy(&f, p, 4);
                    tensor.float_data.push_back(f);
                    p += 4;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 7: { // int64_data (repeated int64, packed or non-packed)
                if (wire_type == 2) { // packed
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    const uint8_t* sub_end = p + len;
                    while (p < sub_end) {
                        tensor.int64_data.push_back((int64_t)read_varint(p, sub_end));
                    }
                } else if (wire_type == 0) { // non-packed
                    tensor.int64_data.push_back((int64_t)read_varint(p, end));
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 8: { // name (string)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    tensor.name.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 9: { // raw_data (bytes)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    tensor.raw_data.assign(p, p + len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            default:
                skip_field(p, end, wire_type);
                break;
        }
    }

    // 濡傛灉鏈塺aw_data涓旀槸float绫诲瀷锛岃浆鎹?
    if (tensor.data_type == TENSOR_FLOAT && !tensor.raw_data.empty() && tensor.float_data.empty()) {
        size_t count = tensor.raw_data.size() / 4;
        tensor.float_data.resize(count);
        memcpy(tensor.float_data.data(), tensor.raw_data.data(), count * 4);
    }
}

// 解析AttributeProto
static inline void parse_attribute(const uint8_t* p, const uint8_t* end, OnnxNode::Attribute& attr) {
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        switch (field_num) {
            case 1: { // name (string)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    attr.name.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 20: { // type (enum, int32)
                if (wire_type == 0) {
                    attr.type = (int32_t)read_varint(p, end);
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 2: { // f (float) - fixed32
                if (wire_type == 5) {
                    float f;
                    memcpy(&f, p, 4);
                    attr.floats.push_back(f);
                    p += 4;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 3: { // i (int64)
                if (wire_type == 0) {
                    attr.ints.push_back((int64_t)read_varint(p, end));
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 4: { // s (string/bytes)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    attr.s.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 5: { // t (TensorProto)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    parse_tensor_proto(p, p + len, attr.t);
                    attr.has_t = true;
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 6: // g (GraphProto) - skip
            case 14: // tp - skip
            case 22: // sparse_tensor - skip
                skip_field(p, end, wire_type);
                break;
            case 7: { // floats (repeated float, packed or non-packed)
                if (wire_type == 2) { // packed
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    size_t count = len / 4;
                    for (size_t i = 0; i < count; i++) {
                        float f;
                        memcpy(&f, p + i * 4, 4);
                        attr.floats.push_back(f);
                    }
                    p += len;
                } else if (wire_type == 5) { // non-packed
                    float f;
                    memcpy(&f, p, 4);
                    attr.floats.push_back(f);
                    p += 4;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 8: { // ints (repeated int64, packed or non-packed)
                if (wire_type == 2) { // packed
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    const uint8_t* sub_end = p + len;
                    while (p < sub_end) {
                        attr.ints.push_back((int64_t)read_varint(p, sub_end));
                    }
                } else if (wire_type == 0) { // non-packed
                    attr.ints.push_back((int64_t)read_varint(p, end));
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 9: { // strings (repeated bytes)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    p += len; // skip
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            default:
                skip_field(p, end, wire_type);
                break;
        }
    }
}

// 瑙ｆ瀽NodeProto
static inline void parse_node_proto(const uint8_t* p, const uint8_t* end, OnnxNode& node) {
    const uint8_t* orig_p = p;
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        switch (field_num) {
            case 1: { // input (repeated string)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    node.inputs.emplace_back((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 2: { // output (repeated string)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    node.outputs.emplace_back((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 3: { // name
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    node.name.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 4: { // op_type
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    node.op_type.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 5: { // attribute (repeated AttributeProto)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    OnnxNode::Attribute attr;
                    const uint8_t* attr_end = p + len;
                    parse_attribute(p, attr_end, attr);
                    if (p != attr_end) p = attr_end; // 容错：强制修正指针
                    node.attributes.push_back(std::move(attr));
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 6: { // doc_string
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            default:
                skip_field(p, end, wire_type);
                break;
        }
    }
}

// 瑙ｆ瀽TypeProto锛堝彧鎻愬彇shape鍜宒ata_type锛?
static inline void parse_type_proto(const uint8_t* p, const uint8_t* end, OnnxValueInfo& vi) {
    // TypeProto -> tensor_type (field 1) -> elem_type (field 1), shape (field 2)
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        if (field_num == 1 && wire_type == WIRE_LENGTH_DELIMITED) {
            uint64_t len = read_varint(p, end);
            const uint8_t* tt_end = p + len;
            // TypeProto.Tensor
            while (p < tt_end) {
                uint32_t wt2;
                uint32_t fn2 = read_tag(p, tt_end, wt2);
                if (fn2 == 1) { // elem_type
                    vi.data_type = (int32_t)read_varint(p, tt_end);
                } else if (fn2 == 2 && wt2 == WIRE_LENGTH_DELIMITED) { // shape
                    uint64_t slen = read_varint(p, tt_end);
                    const uint8_t* s_end = p + slen;
                    // TensorShapeProto -> dim (field 1, repeated)
                    while (p < s_end) {
                        uint32_t wt3;
                        uint32_t fn3 = read_tag(p, s_end, wt3);
                        if (fn3 == 1 && wt3 == WIRE_LENGTH_DELIMITED) {
                            uint64_t dlen = read_varint(p, s_end);
                            const uint8_t* d_end = p + dlen;
                            // Dimension -> dim_value (field 1) or dim_param (field 2)
                            while (p < d_end) {
                                uint32_t wt4;
                                uint32_t fn4 = read_tag(p, d_end, wt4);
                                if (fn4 == 1) {
                                    vi.dims.push_back((int64_t)read_varint(p, d_end));
                                } else {
                                    skip_field(p, d_end, wt4);
                                }
                            }
                        } else {
                            skip_field(p, s_end, wt3);
                        }
                    }
                } else {
                    skip_field(p, tt_end, wt2);
                }
            }
        } else {
            skip_field(p, end, wire_type);
        }
    }
}

// 解析ValueInfoProto
static inline void parse_value_info(const uint8_t* p, const uint8_t* end, OnnxValueInfo& vi) {
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        switch (field_num) {
            case 1: { // name
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    vi.name.assign((const char*)p, len);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 2: { // type (TypeProto)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    if (len > (uint64_t)(end - p)) len = (uint64_t)(end - p);
                    parse_type_proto(p, p + len, vi);
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            default:
                skip_field(p, end, wire_type);
                break;
        }
    }
}

// 瑙ｆ瀽GraphProto
static inline void parse_graph_proto(const uint8_t* p, const uint8_t* end, OnnxModel& model) {
    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        switch (field_num) {
            case 1: { // node (repeated NodeProto)
                uint64_t len = read_varint(p, end);
                OnnxNode node;
                parse_node_proto(p, p + len, node);
                p += len;
                model.nodes.push_back(std::move(node));
                break;
            }
            case 2: { // name
                uint64_t len = read_varint(p, end);
                p += len;
                break;
            }
            case 5: { // initializer (repeated TensorProto)
                if (wire_type == 2) {
                    uint64_t len = read_varint(p, end);
                    OnnxTensor tensor;
                    parse_tensor_proto(p, p + len, tensor);
                    model.initializers.push_back(std::move(tensor));
                    p += len;
                } else {
                    skip_field(p, end, wire_type);
                }
                break;
            }
            case 11: { // input (repeated ValueInfoProto)
                uint64_t len = read_varint(p, end);
                OnnxValueInfo vi;
                parse_value_info(p, p + len, vi);
                model.inputs.push_back(std::move(vi));
                p += len;
                break;
            }
            case 12: { // output (repeated ValueInfoProto)
                uint64_t len = read_varint(p, end);
                OnnxValueInfo vi;
                parse_value_info(p, p + len, vi);
                model.outputs.push_back(std::move(vi));
                p += len;
                break;
            }
            case 13: { // value_info (repeated ValueInfoProto)
                uint64_t len = read_varint(p, end);
                OnnxValueInfo vi;
                parse_value_info(p, p + len, vi);
                model.value_info.push_back(std::move(vi));
                p += len;
                break;
            }
            default:
                skip_field(p, end, wire_type);
                break;
        }
    }
}

// 瑙ｆ瀽ModelProto
static inline OnnxModel parse_onnx(const uint8_t* data, size_t size) {
    OnnxModel model;
    const uint8_t* p = data;
    const uint8_t* end = data + size;

    while (p < end) {
        uint32_t wire_type;
        uint32_t field_num = read_tag(p, end, wire_type);
        if (field_num == 7 && wire_type == WIRE_LENGTH_DELIMITED) { // graph
            uint64_t len = read_varint(p, end);
            parse_graph_proto(p, p + len, model);
            p += len;
        } else {
            skip_field(p, end, wire_type);
        }
    }

    return model;
}

// 浠庢枃浠跺姞杞絆NNX妯″瀷
static inline OnnxModel load_onnx_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) throw std::runtime_error("cannot open file");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(size);
    fread(data.data(), 1, size, f);
    fclose(f);
    return parse_onnx(data.data(), data.size());
}

} // namespace onnx_lite


