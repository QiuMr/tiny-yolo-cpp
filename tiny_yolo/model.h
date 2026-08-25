// Tiny YOLO - 模型加载和图执行引擎
#pragma once
#include "tensor.h"
#include "ops.h"
#include "model_format.h"
#include "onnx_lite.h"
#include "onnx_convert.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>

struct OpInstance {
    uint8_t op_type;
    std::vector<uint16_t> input_ids;
    std::vector<uint16_t> output_ids;
    std::vector<uint8_t> attr_data;
};

struct TensorInfo {
    uint16_t id;
    uint8_t ndim;
    uint8_t dtype;
    uint8_t is_const;
    uint32_t data_offset;
    std::vector<int> shape;
};

class TinyModel {
public:
    std::vector<TensorInfo> tensor_infos;
    std::vector<OpInstance> ops;
    std::vector<Tensor> tensors;       // 鎵€鏈夊紶閲忥紙杩愯鏃讹級
    std::vector<uint8_t> weights_data; // 甯搁噺鏉冮噸鏁版嵁鍖?
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::unordered_map<std::string, uint16_t> name_to_id;
    bool loaded = false;

    ~TinyModel() { release(); }

    void release() {
        tensors.clear();
        weights_data.clear();
        tensor_infos.clear();
        ops.clear();
        loaded = false;
        clear_winograd_weight_cache(); // 权重内存即将释放，清理 Winograd 权重缓存
        clear_packed_W_cache();        // 同理清理 1x1 打包权重缓存（防止地址复用导致脏数据）
        SimpleThreadPool::instance().shutdown();
    }

    // 初始化线程池（32/64 位都启用；32 位限制线程数以节约地址空间）
    static void init_thread_pool() {
        int nth = (int)std::thread::hardware_concurrency();
#ifdef _WIN64
        nth = std::min(nth, 16);
#else
        nth = std::min(nth, 12);
#endif
        nth = std::max(1, nth);
        {
            static const int forced = []() {
                const char* e = getenv("TINY_YOLO_THREADS");
                return e ? atoi(e) : 0;
            }();
            if (forced > 0) nth = forced;
        }
        SimpleThreadPool::instance().init(nth);
    }

    bool load_from_file(const char* path) {
        init_thread_pool();

        FILE* f = fopen(path, "rb");
        if (!f) return false;

        // 璇诲ご閮?
        TyroHeader header;
        if (fread(&header, sizeof(header), 1, f) != 1) { fclose(f); return false; }
        if (header.magic != TYO_MAGIC) { fclose(f); return false; }

        // 璇昏緭鍏ュ悕
        input_names.resize(header.input_count);
        for (int i = 0; i < header.input_count; i++) {
            uint16_t len;
            fread(&len, sizeof(len), 1, f);
            input_names[i].resize(len);
            fread(&input_names[i][0], 1, len, f);
        }

        // 璇昏緭鍑哄悕
        output_names.resize(header.output_count);
        for (int i = 0; i < header.output_count; i++) {
            uint16_t len;
            fread(&len, sizeof(len), 1, f);
            output_names[i].resize(len);
            fread(&output_names[i][0], 1, len, f);
        }

        // 璇诲紶閲忚〃
        tensor_infos.resize(header.tensor_count);
        for (int i = 0; i < header.tensor_count; i++) {
            TensorInfo& ti = tensor_infos[i];
            // 璇诲浐瀹氶儴鍒?(TensorDesc: id(2) + ndim(1) + dtype(1) + is_const(1) + reserved(3) + offset(4) = 12)
            uint16_t id; uint8_t ndim, dtype, is_const, r1, r2, r3; uint32_t offset;
            fread(&id, 2, 1, f);
            fread(&ndim, 1, 1, f);
            fread(&dtype, 1, 1, f);
            fread(&is_const, 1, 1, f);
            fread(&r1, 1, 1, f);
            fread(&r2, 1, 1, f);
            fread(&r3, 1, 1, f);
            fread(&offset, 4, 1, f);
            ti.id = id; ti.ndim = ndim; ti.dtype = dtype;
            ti.is_const = is_const; ti.data_offset = offset;
            ti.shape.resize(ndim);
            for (int d = 0; d < ndim; d++) {
                int32_t v;
                fread(&v, 4, 1, f);
                ti.shape[d] = v;
            }
        }

        // 璇荤畻瀛愯〃
        ops.resize(header.op_count);
        for (int i = 0; i < header.op_count; i++) {
            OpInstance& op = ops[i];
            uint8_t optype, incnt, outcnt, attrsize;
            fread(&optype, 1, 1, f);
            fread(&incnt, 1, 1, f);
            fread(&outcnt, 1, 1, f);
            fread(&attrsize, 1, 1, f);
            op.op_type = optype;
            op.attr_data.resize(attrsize);
            if (attrsize > 0) fread(op.attr_data.data(), 1, attrsize, f);
            op.input_ids.resize(incnt);
            for (int j = 0; j < incnt; j++) { uint16_t v; fread(&v, 2, 1, f); op.input_ids[j] = v; }
            op.output_ids.resize(outcnt);
            for (int j = 0; j < outcnt; j++) { uint16_t v; fread(&v, 2, 1, f); op.output_ids[j] = v; }
        }

        // 璇绘潈閲嶆暟鎹?
        weights_data.resize(header.weights_size);
        if (header.weights_size > 0) {
            fread(weights_data.data(), 1, header.weights_size, f);
        }

        fclose(f);

        // 初始化运行时张量
        tensors.resize(tensor_infos.size());
        for (size_t i = 0; i < tensor_infos.size(); i++) {
            TensorInfo& ti = tensor_infos[i];
            // 统一 dtype 到 ONNX 原生编码（FLOAT=1 / INT32=6 / INT64=7）：
            // 旧版 .tyro 用 0=float / 1=int64 / 2=int32，这里重映射，
            // 之后所有算子只按 ONNX 编码解析（7=int64、6=int32、其余按 float）
            if (ti.dtype == 1) ti.dtype = 7;
            else if (ti.dtype == 2) ti.dtype = 6;
            tensors[i].dtype = ti.dtype;
            if (ti.is_const) {
                // 常量张量引用权重数据区
                float* ptr = (float*)(weights_data.data() + ti.data_offset);
                tensors[i].reference(ti.shape, ptr);
            } else {
                // 非常量张量：设置形状（但不分配内存，算子执行时分配）
                tensors[i].shape = ti.shape;
                tensors[i].ndim = (int)ti.shape.size();
                tensors[i].numel = 1;
                for (int s : ti.shape) tensors[i].numel *= s;
                tensors[i].data = nullptr;
                tensors[i].own_data = false;
            }
        }

        loaded = true;
        return true;
    }

    // 从ONNX文件加载模型
    bool load_from_onnx(const char* path) {
        release();

        init_thread_pool();

        // 1. 读取文件
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "Cannot open %s\n", path); return false; }
        fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<uint8_t> data(fsz);
        fread(data.data(), 1, fsz, f);
        fclose(f);

        // 2. 解析ONNX
        onnx_lite::OnnxModel onnx_model = onnx_lite::parse_onnx(data.data(), data.size());
        fprintf(stderr, "ONNX: %zu nodes, %zu initializers\n",
            onnx_model.nodes.size(), onnx_model.initializers.size());

        // 2.1 图优化：融合Conv+SiLU和Softmax
        int fused_silu = fuse_conv_silu(onnx_model.nodes);
        int fused_softmax = fuse_softmax(onnx_model.nodes);
        fprintf(stderr, "Fused: %d ConvSiLU, %d Softmax, %zu nodes remaining\n",
            fused_silu, fused_softmax, onnx_model.nodes.size());

        // 3. 建立tensor name到id的映射
        std::unordered_map<std::string, uint16_t> n2id;
        uint16_t next_id = 0;

        // 输入
        for (auto& inp : onnx_model.inputs) {
            if (n2id.find(inp.name) == n2id.end()) {
                n2id[inp.name] = next_id++;
                input_names.push_back(inp.name);
            }
        }
        // 初始值（常量）
        for (auto& init : onnx_model.initializers) {
            if (n2id.find(init.name) == n2id.end()) {
                n2id[init.name] = next_id++;
            }
        }
        // 节点输出
        for (auto& node : onnx_model.nodes) {
            for (auto& o : node.outputs) {
                if (!o.empty() && n2id.find(o) == n2id.end()) {
                    n2id[o] = next_id++;
                }
            }
        }
        // 输出
        for (auto& out : onnx_model.outputs) {
            if (n2id.find(out.name) == n2id.end()) {
                n2id[out.name] = next_id++;
            }
            output_names.push_back(out.name);
        }

        // 4. 构建tensor_infos和weights_data
        tensor_infos.resize(next_id);
        for (size_t i = 0; i < tensor_infos.size(); i++) {
            tensor_infos[i].id = (uint16_t)i;
            tensor_infos[i].ndim = 0;
            tensor_infos[i].dtype = 0;
            tensor_infos[i].is_const = 0;
            tensor_infos[i].data_offset = 0;
        }

        // 输入张量形状
        for (auto& inp : onnx_model.inputs) {
            uint16_t id = n2id[inp.name];
            TensorInfo& ti = tensor_infos[id];
            ti.ndim = (uint8_t)inp.dims.size();
            ti.shape.resize(inp.dims.size());
            for (size_t d = 0; d < inp.dims.size(); d++) {
                ti.shape[d] = (int)inp.dims[d];
                if (ti.shape[d] <= 0) ti.shape[d] = 1;
            }
        }

        // 常量张量
        for (auto& init : onnx_model.initializers) {
            uint16_t id = n2id[init.name];
            TensorInfo& ti = tensor_infos[id];
            ti.id = id;
            ti.ndim = (uint8_t)init.dims.size();
            ti.dtype = (uint8_t)init.data_type;  // 使用正确的data_type
            ti.is_const = 1;
            ti.data_offset = (uint32_t)weights_data.size();
            ti.shape.resize(init.dims.size());
            for (size_t d = 0; d < init.dims.size(); d++) {
                ti.shape[d] = (int)init.dims[d];
            }
            size_t numel = 1;
            for (int s : ti.shape) numel *= s;
            size_t old = weights_data.size();

            // 根据数据类型分配内存
            size_t elem_size = 4;  // 默认float
            if (init.data_type == onnx_lite::TENSOR_INT64) elem_size = 8;
            else if (init.data_type == onnx_lite::TENSOR_INT32) elem_size = 4;

            weights_data.resize(old + numel * elem_size);

            if (!init.float_data.empty()) {
                memcpy(weights_data.data() + old, init.float_data.data(), numel * 4);
            } else if (!init.int64_data.empty()) {
                memcpy(weights_data.data() + old, init.int64_data.data(), numel * 8);
            } else if (!init.raw_data.empty()) {
                size_t copy_size = std::min(numel * elem_size, init.raw_data.size());
                memcpy(weights_data.data() + old, init.raw_data.data(), copy_size);
            }
        }

        // 中间张量
        for (auto& node : onnx_model.nodes) {
            for (auto& o : node.outputs) {
                if (o.empty()) continue;
                uint16_t id = n2id[o];
                TensorInfo& ti = tensor_infos[id];
                if (ti.id == 0 && ti.ndim == 0 && !ti.is_const) {
                    ti.id = id;
                    ti.dtype = 0;
                    ti.is_const = 0;
                    ti.data_offset = 0;
                }
            }
        }

        // 从value_info中获取中间张量的形状
        int vi_count = 0;
        for (auto& vi : onnx_model.value_info) {
            auto it = n2id.find(vi.name);
            if (it != n2id.end() && !vi.dims.empty()) {
                TensorInfo& ti = tensor_infos[it->second];
                ti.ndim = (uint8_t)vi.dims.size();
                ti.shape.resize(vi.dims.size());
                for (size_t d = 0; d < vi.dims.size(); d++) {
                    ti.shape[d] = (int)vi.dims[d];
                    if (ti.shape[d] <= 0) ti.shape[d] = 1;
                }
                vi_count++;
            }
        }

        // 5. 构建ops
        for (auto& node : onnx_model.nodes) {
            // 特殊处理Constant算子：将其value作为常量张量初始化，不添加到ops中
            if (node.op_type == "Constant") {
                for (auto& a : node.attributes) {
                    if (a.name == "value" && a.has_t) {
                        for (auto& out : node.outputs) {
                            if (out.empty()) continue;
                            auto it = n2id.find(out);
                            if (it == n2id.end()) continue;
                            uint16_t tid = it->second;
                            TensorInfo& ti = tensor_infos[tid];
                            ti.is_const = 1;
                            ti.dtype = (uint8_t)a.t.data_type;
                            ti.ndim = (uint8_t)a.t.dims.size();
                            ti.shape.resize(a.t.dims.size());
                            for (size_t d = 0; d < a.t.dims.size(); d++) {
                                ti.shape[d] = (int)a.t.dims[d];
                            }
                            size_t numel = 1;
                            for (int s : ti.shape) numel *= s;
                            // 统一存储为float（4字节），int64转换为float
                            ti.dtype = 0; // FLOAT
                            size_t elem_size = 4;
                            ti.data_offset = (uint32_t)weights_data.size();
                            size_t old = weights_data.size();
                            weights_data.resize(old + numel * elem_size);
                            float* dst = (float*)(weights_data.data() + old);
                            if (!a.t.float_data.empty()) {
                                for (size_t i = 0; i < numel && i < a.t.float_data.size(); i++) dst[i] = a.t.float_data[i];
                            } else if (!a.t.int64_data.empty()) {
                                for (size_t i = 0; i < numel && i < a.t.int64_data.size(); i++) dst[i] = (float)a.t.int64_data[i];
                            } else if (!a.t.raw_data.empty()) {
                                if (a.t.data_type == onnx_lite::TENSOR_INT64) {
                                    const int64_t* src = (const int64_t*)a.t.raw_data.data();
                                    for (size_t i = 0; i < numel; i++) dst[i] = (float)src[i];
                                } else {
                                    memcpy(dst, a.t.raw_data.data(), std::min(numel * 4, a.t.raw_data.size()));
                                }
                            }
                        }
                    }
                }
                continue;  // 不添加到ops中
            }

            OpInstance op;
            op.op_type = map_op_type(node.op_type);

            // 输入ID（跳过空字符串）
            for (auto& inp : node.inputs) {
                if (inp.empty()) continue;
                auto it = n2id.find(inp);
                if (it != n2id.end()) {
                    op.input_ids.push_back(it->second);
                } else {
                    fprintf(stderr, "Warning: Unknown input '%s' for op '%s'\n", inp.c_str(), node.op_type.c_str());
                }
            }

            // 输出ID
            for (auto& out : node.outputs) {
                if (out.empty()) continue;
                auto it = n2id.find(out);
                if (it != n2id.end()) {
                    op.output_ids.push_back(it->second);
                }
            }

            // 属性转换
            if (op.op_type == OP_Conv || op.op_type == OP_ConvSiLU) {
                // Conv属性直接序列化为ConvAttr结构体
                ConvAttr attr = {0};
                attr.kernel_h = 3; attr.kernel_w = 3;
                attr.stride_h = 1; attr.stride_w = 1;
                attr.pad_h = 0; attr.pad_w = 0;
                attr.dilation_h = 1; attr.dilation_w = 1;
                attr.group = 1;

                for (auto& a : node.attributes) {
                    if (a.name == "kernel_shape" && a.ints.size() >= 2) {
                        attr.kernel_h = a.ints[0];
                        attr.kernel_w = a.ints[1];
                    } else if (a.name == "strides" && a.ints.size() >= 2) {
                        attr.stride_h = a.ints[0];
                        attr.stride_w = a.ints[1];
                    } else if (a.name == "pads" && a.ints.size() >= 4) {
                        attr.pad_h = a.ints[0];
                        attr.pad_w = a.ints[1];
                    } else if (a.name == "dilations" && a.ints.size() >= 2) {
                        attr.dilation_h = a.ints[0];
                        attr.dilation_w = a.ints[1];
                    } else if (a.name == "group") {
                        if (!a.ints.empty()) attr.group = (int)a.ints[0];
                    }
                }

                op.attr_data.resize(sizeof(ConvAttr));
                memcpy(op.attr_data.data(), &attr, sizeof(ConvAttr));

                // 调试信息已移除
            } else {
                // 其他算子属性转换为TLV格式
                convert_attrs(node, op.attr_data);
            }

            ops.push_back(std::move(op));
        }

        fprintf(stderr, "Built: %zu tensors, %zu ops, %zu bytes weights\n",
            tensor_infos.size(), ops.size(), weights_data.size());

        // 6. 初始化运行时张量
        tensors.resize(tensor_infos.size());
        for (size_t i = 0; i < tensor_infos.size(); i++) {
            TensorInfo& ti = tensor_infos[i];
            tensors[i].dtype = ti.dtype;
            if (ti.is_const) {
                float* ptr = (float*)(weights_data.data() + ti.data_offset);
                tensors[i].reference(ti.shape, ptr);
            } else {
                tensors[i].shape = ti.shape;
                tensors[i].ndim = (int)ti.shape.size();
                tensors[i].numel = 1;
                for (int s : ti.shape) tensors[i].numel *= s;
                tensors[i].data = nullptr;
                tensors[i].own_data = false;
            }
        }

        loaded = true;
        return true;
    }

    // 设置输入张量数据
    bool set_input(int idx, const std::vector<int>& shape, const float* data) {
        if (idx >= (int)input_names.size()) return false;
        uint16_t tid = (uint16_t)idx;
        tensors[tid].reference(shape, (float*)data);
        tensors[tid].own_data = false;
        return true;
    }

    // 鑾峰彇杈撳嚭寮犻噺鏁版嵁
    const Tensor* get_output(int idx) {
        if (idx >= (int)output_names.size()) return nullptr;
        // 杈撳嚭寮犻噺閫氬父鍦ㄥ紶閲忚〃鐨勫悗闈紝鎸夊悕瀛楁煡鎵?
        // 绠€鍖栵細閬嶅巻鎵炬渶鍚庡嚑涓潪甯搁噺寮犻噺
        // 瀹為檯涓婃垜浠渶瑕?name_to_id 鏄犲皠锛屼絾杞崲鏃舵病瀛樺悕瀛?
        // 鐢ㄨ緭鍑洪『搴忥細杈撳嚭寮犻噺鏄浘鐨勬渶鍚庤緭鍑?
        // 鎵炬渶鍚庝竴涓畻瀛愮殑杈撳嚭
        if (ops.empty()) return nullptr;
        uint16_t tid = ops.back().output_ids[idx];
        return &tensors[tid];
    }

    // 鎵ц鎺ㄧ悊
    bool run() {
        if (!loaded) return false;

        // 性能分析仅在设置 TINY_YOLO_PROFILE 环境变量时开启（避免每帧开销）
        static const bool prof_enabled = (getenv("TINY_YOLO_PROFILE") != nullptr);
        // NaN/Inf 追踪：TINY_YOLO_NAN_DEBUG=1 时逐算子检查输出，定位数值爆炸源头
        static const bool nan_debug = (getenv("TINY_YOLO_NAN_DEBUG") != nullptr);
        double op_time[256] = {0};
        int op_count[256] = {0};

        for (size_t i = 0; i < ops.size(); i++) {
            OpInstance& op = ops[i];
            auto t0 = std::chrono::high_resolution_clock::now();
            if (!execute_op(op)) {
                printf("Op %zu failed! op_type=%d inputs=%zu outputs=%zu\n", i, op.op_type, op.input_ids.size(), op.output_ids.size());
                for (size_t k = 0; k < op.input_ids.size(); k++) {
                    Tensor& t = tensors[op.input_ids[k]];
                    printf("  input %zu: ndim=%d shape=[", k, t.ndim);
                    for (int d = 0; d < t.ndim; d++) printf("%d,", t.shape[d]);
                    printf("] numel=%d data=%p\n", t.numel, (void*)t.data);
                }
                return false;
            }
            if (prof_enabled) {
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                op_time[op.op_type] += ms;
                op_count[op.op_type]++;
            }
            if (nan_debug && op.output_ids.size() > 0) {
                uint16_t oid = op.output_ids[0];
                Tensor& t = tensors[oid];
                if (t.dtype == 0 && t.data) {
                    size_t nan_cnt = 0, inf_cnt = 0;
                    float mn = 1e30f, mx = -1e30f;
                    for (int k = 0; k < t.numel; k++) {
                        float v = t.data[k];
                        if (_isnan(v)) { nan_cnt++; continue; }
                        if (!_finite(v)) { inf_cnt++; continue; }
                        if (v < mn) mn = v;
                        if (v > mx) mx = v;
                    }
                    if (nan_cnt || inf_cnt) {
                        printf("[NaN] op %zu type=%d -> tensor %u ndim=%d shape=[", i, op.op_type, oid, t.ndim);
                        for (int d = 0; d < t.ndim; d++) printf("%d,", t.shape[d]);
                        printf("] nan=%zu inf=%zu finite_range=[%g,%g]\n", nan_cnt, inf_cnt, mn, mx);
                        // 打印前 8 个 NaN 的线性索引及 4D 解码位置
                        int shown = 0;
                        for (int k = 0; k < t.numel && shown < 8; k++) {
                            if (!_isnan(t.data[k])) continue;
                            printf("    nan_idx[%d] flat=%d", shown, k);
                            if (t.ndim == 4) {
                                int HW = t.shape[2] * t.shape[3];
                                int c = k / HW, rem = k % HW;
                                printf(" -> n0 c%d h%d w%d", c, rem / t.shape[3], rem % t.shape[3]);
                            }
                            printf(" val=%g\n", t.data[k]);
                            shown++;
                        }
                        printf("  input shapes:\n");
                        for (size_t k = 0; k < op.input_ids.size(); k++) {
                            Tensor& in_t = tensors[op.input_ids[k]];
                            printf("    in[%zu] t%u ndim=%d dtype=%d shape=[", k, op.input_ids[k], in_t.ndim, in_t.dtype);
                            for (int d = 0; d < in_t.ndim; d++) printf("%d,", in_t.shape[d]);
                            printf("]");
                            // 顺带检查输入是否已含 NaN
                            if (in_t.dtype == 0 && in_t.data) {
                                size_t in_nan = 0;
                                for (int q = 0; q < in_t.numel; q++) if (_isnan(in_t.data[q])) in_nan++;
                                if (in_nan) printf(" ALREADY_NAN=%zu", in_nan);
                            }
                            printf("\n");
                        }
                        return false; // 找到第一个源头即停，避免刷屏
                    }
                }
            }
        }
        if (prof_enabled) {
            fprintf(stderr, "=== Op Performance ===\n");
            for (int t = 0; t < 256; t++) {
                if (op_count[t] > 0) {
                    fprintf(stderr, "  Op %d: count=%d, total=%.1fms, avg=%.2fms\n",
                        t, op_count[t], op_time[t], op_time[t] / op_count[t]);
                }
            }
        }
        return true;
    }

private:
    bool execute_op(const OpInstance& op) {
        switch (op.op_type) {
            case OP_Conv: {
                ConvAttr attr;
                memset(&attr, 0, sizeof(attr));
                int copy_size = sizeof(attr);
                if (op.attr_data.size() < copy_size) copy_size = op.attr_data.size();
                memcpy(&attr, op.attr_data.data(), copy_size);
                Tensor& input = tensors[op.input_ids[0]];
                Tensor& weight = tensors[op.input_ids[1]];
                Tensor* bias = op.input_ids.size() > 2 ? &tensors[op.input_ids[2]] : nullptr;
                Tensor& output = tensors[op.output_ids[0]];
                op_conv(input, weight, bias, output, attr);
                return true;
            }
            case OP_ConvSiLU: {
                // Conv + SiLU 融合：SiLU 在卷积内部（GEMM 后处理/输出变换）直接完成，
                // 省掉一次对整个输出的读写
                ConvAttr attr;
                memset(&attr, 0, sizeof(attr));
                int copy_size = sizeof(attr);
                if (op.attr_data.size() < copy_size) copy_size = op.attr_data.size();
                memcpy(&attr, op.attr_data.data(), copy_size);
                Tensor& input = tensors[op.input_ids[0]];
                Tensor& weight = tensors[op.input_ids[1]];
                Tensor* bias = op.input_ids.size() > 2 ? &tensors[op.input_ids[2]] : nullptr;
                Tensor& output = tensors[op.output_ids[0]];
                op_conv(input, weight, bias, output, attr, true);
                return true;
            }
            case OP_SiLU: {
                op_silu(tensors[op.input_ids[0]], tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Sigmoid: {
                op_sigmoid(tensors[op.input_ids[0]], tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Relu: {
                op_relu(tensors[op.input_ids[0]], tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Exp: {
                op_exp(tensors[op.input_ids[0]], tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Add:
            case OP_Sub:
            case OP_Mul:
            case OP_Div: {
                char c = (op.op_type == OP_Add) ? '+' : (op.op_type == OP_Sub) ? '-' :
                         (op.op_type == OP_Mul) ? '*' : '/';
                Tensor& a = tensors[op.input_ids[0]];
                Tensor& b = tensors[op.input_ids[1]];
                try {
                    op_elementwise(a, b, tensors[op.output_ids[0]], c);
                } catch (const std::exception& e) {
                    fprintf(stderr, "Elementwise op %c exception: %s\n", c, e.what());
                    return false;
                } catch (...) {
                    fprintf(stderr, "Elementwise op %c unknown exception\n", c);
                    return false;
                }
                return true;
            }
            case OP_Concat: {
                // 瑙ｆ瀽 axis 灞炴€?
                int axis = 0;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2) { // INT
                        int64_t axis64 = 0;
                        memcpy(&axis64, &op.attr_data[p], 8); p += 8;
                        axis = (int)axis64;
                    } else { break; }
                }
                std::vector<Tensor*> inputs;
                for (uint16_t id : op.input_ids) inputs.push_back(&tensors[id]);
                op_concat(inputs, tensors[op.output_ids[0]], axis);
                return true;
            }
            case OP_Resize: {
                Tensor& input = tensors[op.input_ids[0]];
                Tensor& output = tensors[op.output_ids[0]];
                // 解析 mode 属性（默认 nearest，YOLO 上采样通常是 nearest）
                const char* mode = "nearest";
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    if (p + nlen > op.attr_data.size()) break;
                    std::string aname((char*)&op.attr_data[p], nlen); p += nlen;
                    if (p >= op.attr_data.size()) break;
                    uint8_t atype = op.attr_data[p++];
                    if (aname == "mode" && atype == 3) {
                        uint16_t slen; memcpy(&slen, &op.attr_data[p], 2); p += 2;
                        if (slen >= 6 && memcmp(&op.attr_data[p], "linear", 6) == 0) mode = "linear";
                        else if (slen >= 7 && memcmp(&op.attr_data[p], "nearest", 7) == 0) mode = "nearest";
                        p += slen;
                    } else {
                        // 跳过未知属性
                        if (atype == 1 || atype == 2) p += 8;
                        else if (atype == 5 || atype == 7) { uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2 + cnt * 8; }
                        else if (atype == 3) { uint16_t slen; memcpy(&slen, &op.attr_data[p], 2); p += 2 + slen; }
                        else break;
                    }
                }
                // 尝试从 sizes 输入读取输出大小
                int outH = 0, outW = 0;
                int scales_idx = -1, sizes_idx = -1;
                if (op.input_ids.size() == 2) { scales_idx = 1; }
                else if (op.input_ids.size() == 3) { scales_idx = 2; }
                else if (op.input_ids.size() >= 4) { scales_idx = 2; sizes_idx = 3; }
                if (sizes_idx >= 0) {
                    Tensor& sizes = tensors[op.input_ids[sizes_idx]];
                    if (sizes.numel >= 4) {
                        if (sizes.dtype == 7) { outH = (int)((const int64_t*)sizes.data)[2]; outW = (int)((const int64_t*)sizes.data)[3]; }
                        else { outH = (int)sizes.data[2]; outW = (int)sizes.data[3]; }
                    }
                }
                if (outH <= 0 || outW <= 0) {
                    if (scales_idx >= 0) {
                        Tensor& scales = tensors[op.input_ids[scales_idx]];
                        if (scales.numel >= 4 && scales.data[2] > 0 && scales.data[3] > 0) {
                            outH = (int)(input.shape[2] * scales.data[2]);
                            outW = (int)(input.shape[3] * scales.data[3]);
                        }
                    }
                }
                if (outH <= 0 || outW <= 0) {
                    outH = output.ndim >= 3 ? output.shape[2] : 0;
                    outW = output.ndim >= 4 ? output.shape[3] : 0;
                }
                if (outH <= 0 || outW <= 0) { outH = input.shape[2]; outW = input.shape[3]; }
                op_resize(input, output, outH, outW, mode);
                return true;
            }
            case OP_Transpose: {
                // 瑙ｆ瀽 perm 灞炴€?
                std::vector<int> perm;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    if (p + nlen > op.attr_data.size()) break;
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    if (p >= op.attr_data.size()) break;
                    uint8_t atype = op.attr_data[p++];
                    // 兼容新旧格式：整型数组 TLV 类型码旧版=7、新版=5（布局相同）
                    if (name == "perm" && (atype == 5 || atype == 7)) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        for (int i = 0; i < cnt; i++) { int64_t v; memcpy(&v, &op.attr_data[p], 8); perm.push_back((int)v); p += 8; }
                    } else { break; }
                }
                if (perm.empty()) {
                    int nd = tensors[op.input_ids[0]].ndim;
                    for (int i = nd - 1; i >= 0; i--) perm.push_back(i);
                }
                Tensor& inp = tensors[op.input_ids[0]];
                op_transpose(inp, tensors[op.output_ids[0]], perm);
                return true;
            }
            case OP_Reshape: {
                if (op.input_ids.size() > 1) {
                    Tensor& shape_tensor = tensors[op.input_ids[1]];
                    std::vector<int> new_shape(shape_tensor.numel);
                    if (shape_tensor.dtype == 7) {  // INT64
                        const int64_t* sd = (const int64_t*)shape_tensor.data;
                        for (int i = 0; i < shape_tensor.numel; i++) new_shape[i] = (int)sd[i];
                    } else if (shape_tensor.dtype == 6) {  // INT32
                        const int32_t* sd = (const int32_t*)shape_tensor.data;
                        for (int i = 0; i < shape_tensor.numel; i++) new_shape[i] = (int)sd[i];
                    } else {
                        // 默认按int64解析
                        const int64_t* sd = (const int64_t*)shape_tensor.data;
                        for (int i = 0; i < shape_tensor.numel; i++) new_shape[i] = (int)sd[i];
                    }
                    Tensor& input = tensors[op.input_ids[0]];
                    op_reshape(input, tensors[op.output_ids[0]], new_shape);
                }
                return true;
            }
            case OP_MatMul: {
                Tensor& a = tensors[op.input_ids[0]];
                Tensor& b = tensors[op.input_ids[1]];
                op_matmul(a, b, tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Split: {
                int axis = 0;
                std::vector<int> split_sizes;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    if (p + nlen > op.attr_data.size()) break;
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    if (p >= op.attr_data.size()) break;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t axis64 = 0;
                        memcpy(&axis64, &op.attr_data[p], 8);
                        axis = (int)axis64;
                        p += 8;
                    } else if (name == "axis" && atype == 1 && p + 4 <= op.attr_data.size()) {
                        memcpy(&axis, &op.attr_data[p], 4);
                        p += 4;
                    } else if (name == "split" && (atype == 5 || atype == 7) && p + 2 <= op.attr_data.size()) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        for (int i = 0; i < cnt && p + 8 <= op.attr_data.size(); i++) {
                            int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8;
                            split_sizes.push_back((int)v);
                        }
                    } else {
                        // 跳过未知属性
                        if (atype == 1 && p + 4 <= op.attr_data.size()) p += 4;
                        else if (atype == 2 && p + 8 <= op.attr_data.size()) p += 8;
                        else if (atype == 3) {
                            if (p + 1 <= op.attr_data.size()) {
                                uint8_t slen = op.attr_data[p++];
                                p += slen;
                            }
                        } else if (atype == 4) {
                            if (p + 2 <= op.attr_data.size()) {
                                uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2 + cnt * 4;
                            }
                        } else if (atype == 5 || atype == 7) {
                            if (p + 2 <= op.attr_data.size()) {
                                uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2 + cnt * 8;
                            }
                        } else {
                            break;
                        }
                    }
                }
                std::vector<Tensor*> outputs;
                for (uint16_t id : op.output_ids) outputs.push_back(&tensors[id]);
                op_split(tensors[op.input_ids[0]], outputs, axis, split_sizes);
                return true;
            }
            case OP_Slice: {
                // starts, ends, axes, steps 閮芥槸杈撳叆锛堝父閲忥級锛岄€氬父鏄?int64
                std::vector<int> starts, ends, axes, steps;
                auto read_int_tensor = [](Tensor& t, std::vector<int>& out) {
                    if (t.dtype == 7) { const int64_t* p = (const int64_t*)t.data; for (int i = 0; i < t.numel; i++) out.push_back((int)p[i]); }
                    else if (t.dtype == 6) { const int32_t* p = (const int32_t*)t.data; for (int i = 0; i < t.numel; i++) out.push_back((int)p[i]); }
                    else { const int64_t* p = (const int64_t*)t.data; for (int i = 0; i < t.numel; i++) out.push_back((int)p[i]); }
                };
                if (op.input_ids.size() > 1) read_int_tensor(tensors[op.input_ids[1]], starts);
                if (op.input_ids.size() > 2) read_int_tensor(tensors[op.input_ids[2]], ends);
                if (op.input_ids.size() > 3) read_int_tensor(tensors[op.input_ids[3]], axes);
                if (op.input_ids.size() > 4) read_int_tensor(tensors[op.input_ids[4]], steps);
                op_slice(tensors[op.input_ids[0]], tensors[op.output_ids[0]], starts, ends, axes, steps);
                return true;
            }
            case OP_ReduceMax:
            case OP_ReduceSum: {
                std::vector<int> axes;
                int keepdims = 1;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    if (p + nlen > op.attr_data.size()) break;
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    if (p >= op.attr_data.size()) break;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axes" && (atype == 5 || atype == 7)) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        for (int i = 0; i < cnt; i++) { int64_t v; memcpy(&v, &op.attr_data[p], 8); axes.push_back((int)v); p += 8; }
                    } else if (name == "keepdims" && atype == 2) {
                        int64_t kd64 = 0;
                        memcpy(&kd64, &op.attr_data[p], 8); p += 8;
                        keepdims = (int)kd64;
                    } else { break; }
                }
                // ONNX锛歛xes 鍙互鏄睘鎬ф垨绗簩涓緭鍏?
                if (axes.empty() && op.input_ids.size() > 1) {
                    Tensor& axes_tensor = tensors[op.input_ids[1]];
                    if (axes_tensor.dtype == 7) {  // INT64
                        const int64_t* p = (const int64_t*)axes_tensor.data;
                        for (int i = 0; i < axes_tensor.numel; i++) axes.push_back((int)p[i]);
                    } else if (axes_tensor.dtype == 6) {  // INT32
                        const int32_t* p = (const int32_t*)axes_tensor.data;
                        for (int i = 0; i < axes_tensor.numel; i++) axes.push_back((int)p[i]);
                    } else {
                        // 默认按int64解析
                        const int64_t* p = (const int64_t*)axes_tensor.data;
                        for (int i = 0; i < axes_tensor.numel; i++) axes.push_back((int)p[i]);
                    }
                }
                // ONNX 榛樿锛歛xes 涓虹┖鏃?reduce 鎵€鏈夌淮搴?
                if (axes.empty()) {
                    for (int d = 0; d < tensors[op.input_ids[0]].ndim; d++) axes.push_back(d);
                }
                op_reduce(tensors[op.input_ids[0]], tensors[op.output_ids[0]], axes, keepdims != 0, op.op_type == OP_ReduceMax);
                return true;
            }
            case OP_Softmax: {
                int axis = -1;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2) {
                        int64_t axis64 = 0;
                        memcpy(&axis64, &op.attr_data[p], 8); p += 8;
                        axis = (int)axis64;
                    }
                    else { break; }
                }
                op_softmax(tensors[op.input_ids[0]], tensors[op.output_ids[0]], axis);
                return true;
            }
            case OP_MaxPool: {
                int kH=2, kW=2, sH=2, sW=2, pH=0, pW=0;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "kernel_shape" && (atype == 5 || atype == 7)) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        if (cnt >= 1) { int64_t v; memcpy(&v, &op.attr_data[p], 8); kH=(int)v; p+=8; }
                        if (cnt >= 2) { int64_t v; memcpy(&v, &op.attr_data[p], 8); kW=(int)v; p+=8; }
                    } else if (name == "strides" && (atype == 5 || atype == 7)) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        if (cnt >= 1) { int64_t v; memcpy(&v, &op.attr_data[p], 8); sH=(int)v; p+=8; }
                        if (cnt >= 2) { int64_t v; memcpy(&v, &op.attr_data[p], 8); sW=(int)v; p+=8; }
                    } else if (name == "pads" && (atype == 5 || atype == 7)) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        int64_t v[4] = {0,0,0,0};
                        for (int i = 0; i < cnt && i < 4; i++) { memcpy(&v[i], &op.attr_data[p], 8); p += 8; }
                        pH = (int)v[0];  // top
                        pW = (int)v[1];  // left
                    } else {
                        // 跳过未知属性
                        if (atype == 1) p += 4;
                        else if (atype == 2) p += 8;
                        else if (atype == 3) { if (p + 1 <= op.attr_data.size()) { uint8_t slen = op.attr_data[p]; p += 1 + slen; } }
                        else if (atype == 4) { if (p + 1 <= op.attr_data.size()) { uint8_t cnt = op.attr_data[p]; p += 1 + cnt * 4; } }
                        else if (atype == 5) { if (p + 2 <= op.attr_data.size()) { uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2 + cnt * 8; } }
                        else if (atype == 6) { if (p + 1 <= op.attr_data.size()) { uint8_t cnt = op.attr_data[p]; p += 1 + cnt; } }
                        else break;
                    }
                }
                op_maxpool(tensors[op.input_ids[0]], tensors[op.output_ids[0]], kH, kW, sH, sW, pH, pW);
                return true;
            }
            case OP_Identity: {
                tensors[op.output_ids[0]].reference(tensors[op.input_ids[0]].shape, tensors[op.input_ids[0]].data);
                tensors[op.output_ids[0]].own_data = false;
                return true;
            }
            case OP_Unsqueeze: {
                std::vector<int> axes;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axes" && (atype == 5 || atype == 7) && p + 2 <= op.attr_data.size()) {
                        uint16_t cnt; memcpy(&cnt, &op.attr_data[p], 2); p += 2;
                        for (int i = 0; i < cnt && p + 8 <= op.attr_data.size(); i++) {
                            int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8;
                            axes.push_back((int)v);
                        }
                    } else { p = op.attr_data.size(); }
                }
                // 如果axes为空，从第二个输入张量读取
                if (axes.empty() && op.input_ids.size() > 1) {
                    Tensor& idx = tensors[op.input_ids[1]];
                    if (idx.dtype == 7) { // INT64
                        const int64_t* p = (const int64_t*)idx.data;
                        for (int i = 0; i < idx.numel; i++) axes.push_back((int)p[i]);
                    } else {
                        for (int i = 0; i < idx.numel; i++) axes.push_back((int)idx.data[i]);
                    }
                }
                op_unsqueeze(tensors[op.input_ids[0]], tensors[op.output_ids[0]], axes);
                return true;
            }
            case OP_Flatten: {
                int axis = 1;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8;
                        axis = (int)v;
                    } else { p = op.attr_data.size(); }
                }
                op_flatten(tensors[op.input_ids[0]], tensors[op.output_ids[0]], axis);
                return true;
            }
            case OP_Tile: {
                std::vector<int> repeats;
                if (op.input_ids.size() > 1) {
                    Tensor& rep = tensors[op.input_ids[1]];
                    if (rep.dtype == 7) { // INT64
                        const int64_t* p = (const int64_t*)rep.data;
                        for (int i = 0; i < rep.numel; i++) repeats.push_back((int)p[i]);
                    } else {
                        for (int i = 0; i < rep.numel; i++) repeats.push_back((int)rep.data[i]);
                    }
                }
                op_tile(tensors[op.input_ids[0]], tensors[op.output_ids[0]], repeats);
                return true;
            }
            case OP_Gather: {
                int axis = 0;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8;
                        axis = (int)v;
                    } else { p = op.attr_data.size(); }
                }
                op_gather(tensors[op.input_ids[0]], tensors[op.output_ids[0]], tensors[op.input_ids[1]], axis);
                return true;
            }
            case OP_GatherElements: {
                int axis = 0;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8;
                        axis = (int)v;
                    } else { p = op.attr_data.size(); }
                }
                Tensor& inp = tensors[op.input_ids[0]];
                Tensor& idx = tensors[op.input_ids[1]];
                op_gather_elements(inp, tensors[op.output_ids[0]], idx, axis);
                return true;
            }
            case OP_TopK: {
                int k = 1, axis = -1;
                bool largest = true, sorted = true;
                for (size_t p = 0; p < op.attr_data.size(); ) {
                    uint8_t nlen = op.attr_data[p++];
                    std::string name((char*)&op.attr_data[p], nlen); p += nlen;
                    uint8_t atype = op.attr_data[p++];
                    if (name == "axis" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8; axis = (int)v;
                    } else if (name == "largest" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8; largest = v != 0;
                    } else if (name == "sorted" && atype == 2 && p + 8 <= op.attr_data.size()) {
                        int64_t v; memcpy(&v, &op.attr_data[p], 8); p += 8; sorted = v != 0;
                    } else { p = op.attr_data.size(); }
                }
                if (op.input_ids.size() > 1) {
                    Tensor& kt = tensors[op.input_ids[1]];
                    if (kt.dtype == 7) { // INT64
                        k = (int)((const int64_t*)kt.data)[0];
                    } else {
                        k = (int)kt.data[0];
                    }
                }
                Tensor& inp = tensors[op.input_ids[0]];
                op_topk(inp, tensors[op.output_ids[0]], tensors[op.output_ids[1]], k, axis, largest, sorted);
                return true;
            }
            case OP_Mod: {
                op_mod(tensors[op.input_ids[0]], tensors[op.input_ids[1]], tensors[op.output_ids[0]]);
                return true;
            }
            case OP_Cast: {
                op_cast(tensors[op.input_ids[0]], tensors[op.output_ids[0]]);
                return true;
            }
            default:
                return false;
        }
    }
};
