#!/usr/bin/env python3
"""
ONNX → .tyro 模型格式转换器
用于 Tiny YOLO 推理引擎
"""
import struct
import sys
import numpy as np
from collections import defaultdict

try:
    import onnx
    from onnx import numpy_helper
except ImportError:
    print("需要安装 onnx: pip install onnx")
    sys.exit(1)

# ============================================================
# 算子类型枚举（与 model_format.h 一致）
# ============================================================
OP_CONV = 0
OP_SILU = 1
OP_SIGMOID = 2
OP_RELU = 3
OP_ADD = 4
OP_MUL = 5
OP_SUB = 6
OP_DIV = 7
OP_EXP = 8
OP_CONCAT = 9
OP_RESIZE = 10
OP_TRANSPOSE = 11
OP_RESHAPE = 12
OP_MATMUL = 13
OP_SPLIT = 14
OP_SLICE = 15
OP_REDUCEMAX = 16
OP_REDUCESUM = 17
OP_SOFTMAX = 18
OP_MAXPOOL = 19
OP_IDENTITY = 20
OP_CONVSILU = 21

# ONNX op_type → 内部 op_type 映射
OP_TYPE_MAP = {
    'Conv': OP_CONV,
    'SiLU': OP_SILU,
    'Sigmoid': OP_SIGMOID,
    'Relu': OP_RELU,
    'Add': OP_ADD,
    'Mul': OP_MUL,
    'Sub': OP_SUB,
    'Div': OP_DIV,
    'Exp': OP_EXP,
    'Concat': OP_CONCAT,
    'Resize': OP_RESIZE,
    'Transpose': OP_TRANSPOSE,
    'Reshape': OP_RESHAPE,
    'MatMul': OP_MATMUL,
    'Split': OP_SPLIT,
    'Slice': OP_SLICE,
    'ReduceMax': OP_REDUCEMAX,
    'ReduceSum': OP_REDUCESUM,
    'Softmax': OP_SOFTMAX,
    'MaxPool': OP_MAXPOOL,
    'Identity': OP_IDENTITY,
    'ConvSiLU': OP_CONVSILU,
}


# ============================================================
# 1. SiLU 融合：把 Sigmoid + Mul 融合成 SiLU
# ============================================================
def fuse_silu(graph):
    """
    SiLU(x) = x * sigmoid(x)
    ONNX 中通常表示为：
      Sigmoid(input) -> sigmoid_out
      Mul(input, sigmoid_out) -> silu_out
    融合后：SiLU(input) -> silu_out
    """
    nodes = list(graph.node)
    removed = set()
    fused_count = 0

    # 建立输出名→节点的映射
    out_to_node = {}
    for node in nodes:
        for out in node.output:
            out_to_node[out] = node

    for i, node in enumerate(nodes):
        if node.op_type != 'Mul':
            continue
        if i in removed:
            continue

        # 检查两个输入：一个是原始输入，另一个是 Sigmoid 的输出
        input_a, input_b = node.input[0], node.input[1]

        # 情况1: input_a 是 Sigmoid 输出，input_b 是原始输入
        sigmoid_node = None
        original_input = None
        if input_a in out_to_node and out_to_node[input_a].op_type == 'Sigmoid':
            sigmoid_node = out_to_node[input_a]
            original_input = input_b
            sigmoid_input = sigmoid_node.input[0]
        elif input_b in out_to_node and out_to_node[input_b].op_type == 'Sigmoid':
            sigmoid_node = out_to_node[input_b]
            original_input = input_a
            sigmoid_input = sigmoid_node.input[0]
        else:
            continue

        # 确认 Sigmoid 的输入和 Mul 的另一个输入相同
        if sigmoid_input != original_input:
            continue

        # 确认 Sigmoid 只被这个 Mul 使用
        sigmoid_used_by = []
        for n in nodes:
            if sigmoid_node.output[0] in n.input and n != node:
                sigmoid_used_by.append(n)
        if sigmoid_used_by:
            continue

        # 创建融合后的 SiLU 节点
        silu_node = onnx.helper.make_node(
            'SiLU',
            inputs=[original_input],
            outputs=[node.output[0]],
            name=node.name or f"silu_{fused_count}"
        )

        # 替换节点
        nodes[i] = silu_node
        removed.add(nodes.index(sigmoid_node))
        fused_count += 1

    # 移除被融合的 Sigmoid 节点
    new_nodes = [n for i, n in enumerate(nodes) if i not in removed]
    graph.ClearField('node')
    graph.node.extend(new_nodes)

    print(f"SiLU 融合: {fused_count} 个")
    return graph


# ============================================================
# 1.5 Softmax 融合：把 ReduceMax + Sub + Exp + ReduceSum + Div 融合成 Softmax
# ============================================================
def fuse_softmax(graph):
    """
    标准 Softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
    ONNX 中通常表示为：
      ReduceMax(input) -> max_out
      Sub(input, max_out) -> sub_out
      Exp(sub_out) -> exp_out
      ReduceSum(exp_out) -> sum_out
      Div(exp_out, sum_out) -> softmax_out
    融合后：Softmax(input) -> softmax_out
    """
    nodes = list(graph.node)
    removed = set()
    fused_count = 0

    out_to_node = {}
    for node in nodes:
        for out in node.output:
            out_to_node[out] = node

    for i, node in enumerate(nodes):
        if node.op_type != 'Div':
            continue
        if i in removed:
            continue

        # Div 的两个输入：一个是 Exp 的输出，另一个是 ReduceSum 的输出
        div_input_a, div_input_b = node.input[0], node.input[1]

        exp_node = None
        reducesum_node = None

        if div_input_a in out_to_node and out_to_node[div_input_a].op_type == 'Exp':
            exp_node = out_to_node[div_input_a]
        if div_input_b in out_to_node and out_to_node[div_input_b].op_type == 'ReduceSum':
            reducesum_node = out_to_node[div_input_b]

        if not exp_node or not reducesum_node:
            continue

        # Exp 的输入应该是 Sub 的输出
        exp_input = exp_node.input[0]
        if exp_input not in out_to_node or out_to_node[exp_input].op_type != 'Sub':
            continue
        sub_node = out_to_node[exp_input]

        # ReduceSum 的输入应该是 Exp 的输出
        if reducesum_node.input[0] != exp_node.output[0]:
            continue

        # Sub 的两个输入：一个是原始输入，另一个是 ReduceMax 的输出
        sub_input_a, sub_input_b = sub_node.input[0], sub_node.input[1]
        original_input = None
        reducemax_node = None

        if sub_input_a in out_to_node and out_to_node[sub_input_a].op_type == 'ReduceMax':
            reducemax_node = out_to_node[sub_input_a]
            original_input = sub_input_b
        elif sub_input_b in out_to_node and out_to_node[sub_input_b].op_type == 'ReduceMax':
            reducemax_node = out_to_node[sub_input_b]
            original_input = sub_input_a
        else:
            continue

        # 确认 ReduceMax 的输入和 Sub 的另一个输入相同
        if reducemax_node.input[0] != original_input:
            continue

        # 确认中间节点只被融合链内的节点使用
        def has_external_consumer(out_name, internal_list):
            for n in nodes:
                if n not in internal_list and out_name in n.input:
                    return True
            return False

        internal = [reducemax_node, sub_node, exp_node, reducesum_node, node]
        if has_external_consumer(reducemax_node.output[0], internal):
            continue
        if has_external_consumer(sub_node.output[0], internal):
            continue
        if has_external_consumer(exp_node.output[0], internal):
            continue
        if has_external_consumer(reducesum_node.output[0], internal):
            continue

        # 创建融合后的 Softmax 节点
        softmax_node = onnx.helper.make_node(
            'Softmax',
            inputs=[original_input],
            outputs=[node.output[0]],
            name=node.name or f"softmax_{fused_count}"
        )

        # 替换 Div 节点为 Softmax
        nodes[i] = softmax_node
        # 标记被融合的节点
        for n in [reducemax_node, sub_node, exp_node, reducesum_node]:
            idx = nodes.index(n)
            removed.add(idx)
        fused_count += 1

    new_nodes = [n for i, n in enumerate(nodes) if i not in removed]
    graph.ClearField('node')
    graph.node.extend(new_nodes)

    print(f"Softmax 融合: {fused_count} 个")
    return graph


# ============================================================
# 1.8 Conv+SiLU 融合：把 Conv -> Sigmoid -> Mul 融合成 ConvSiLU
# ============================================================
def fuse_conv_silu(graph):
    """
    识别 Conv -> SiLU 模式，融合成 ConvSiLU
    （SiLU已经是融合后的算子，所以直接找Conv后面跟着SiLU）
    """
    nodes = list(graph.node)
    removed = set()
    fused_count = 0

    for i, node in enumerate(nodes):
        if node.op_type != 'Conv':
            continue
        if i in removed:
            continue

        conv_out = node.output[0]

        # 找 SiLU 消费者
        silu_idx = -1
        for j, n in enumerate(nodes):
            if j in removed:
                continue
            if n.op_type == 'SiLU' and conv_out in n.input:
                silu_idx = j
                break

        if silu_idx < 0:
            continue

        silu_node = nodes[silu_idx]

        # 确认 Conv 的输出没有其他消费者
        conv_consumers = [j for j, n in enumerate(nodes)
                          if j != silu_idx and conv_out in n.input]
        if conv_consumers:
            continue

        # 创建融合后的 ConvSiLU 节点
        conv_silu_node = onnx.helper.make_node(
            'ConvSiLU',
            inputs=list(node.input),
            outputs=[silu_node.output[0]],
            name=node.name or f"convsilu_{fused_count}"
        )
        for attr in node.attribute:
            conv_silu_node.attribute.append(attr)

        nodes[i] = conv_silu_node
        removed.add(silu_idx)
        fused_count += 1

    new_nodes = [n for i, n in enumerate(nodes) if i not in removed]
    graph.ClearField('node')
    graph.node.extend(new_nodes)

    print(f"Conv+SiLU 融合: {fused_count} 个")
    return graph


# ============================================================
# 2. 收集常量张量
# ============================================================
def collect_constants(graph):
    constants = {}
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        constants[init.name] = arr
    return constants


# ============================================================
# 3. 拓扑排序（Kahn 算法）
# ============================================================
def topo_sort(graph, constants):
    nodes = list(graph.node)
    n = len(nodes)

    # 建立输出名→节点索引的映射
    out_to_idx = {}
    for i, node in enumerate(nodes):
        for out in node.output:
            out_to_idx[out] = i

    # 计算每个节点的入度（依赖的节点数）
    in_degree = [0] * n
    dependents = defaultdict(set)  # 修复：使用 set 去重

    for i, node in enumerate(nodes):
        deps = set()
        for inp in node.input:
            if inp in out_to_idx:
                deps.add(out_to_idx[inp])
        in_degree[i] = len(deps)
        for d in deps:
            dependents[d].add(i)

    # Kahn 算法
    queue = [i for i in range(n) if in_degree[i] == 0]
    sorted_nodes = []

    while queue:
        idx = queue.pop(0)
        sorted_nodes.append(nodes[idx])
        for dep in dependents.get(idx, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(sorted_nodes) != n:
        print(f"警告: 拓扑排序不完整，{len(sorted_nodes)}/{n}")

    return sorted_nodes


# ============================================================
# 4. 形状推断
# ============================================================
def infer_shapes(sorted_nodes, constants, input_name, input_shape):
    shapes = {input_name: list(input_shape)}

    for node in sorted_nodes:
        op = node.op_type
        inputs = list(node.input)
        outputs = list(node.output)

        # 获取输入形状
        in_shapes = []
        for inp in inputs:
            if inp in shapes:
                in_shapes.append(shapes[inp])
            elif inp in constants:
                in_shapes.append(list(constants[inp].shape))
            else:
                in_shapes.append(None)

        out_shape = None

        try:
            if op == 'Conv' or op == 'ConvSiLU':
                if in_shapes[0] and len(in_shapes[0]) == 4:
                    N, C_in, H, W = in_shapes[0]
                    # 从属性读取
                    kH = kW = 1
                    sH = sW = 1
                    pH = pW = 0
                    dH = dW = 1
                    group = 1
                    for attr in node.attribute:
                        if attr.name == 'kernel_shape':
                            kH, kW = list(attr.ints)
                        elif attr.name == 'strides':
                            sH, sW = list(attr.ints)
                        elif attr.name == 'pads':
                            pH, pW, _, _ = list(attr.ints)
                        elif attr.name == 'dilations':
                            dH, dW = list(attr.ints)
                        elif attr.name == 'group':
                            group = attr.i
                    # 权重形状
                    if in_shapes[1]:
                        C_out = in_shapes[1][0]
                    else:
                        C_out = -1
                    outH = (H + 2*pH - dH*(kH-1) - 1) // sH + 1
                    outW = (W + 2*pW - dW*(kW-1) - 1) // sW + 1
                    out_shape = [N, C_out, outH, outW]

            elif op in ('SiLU', 'Sigmoid', 'Relu', 'Identity'):
                out_shape = in_shapes[0]

            elif op in ('Add', 'Mul', 'Sub', 'Div'):
                # 广播
                if in_shapes[0] and in_shapes[1]:
                    shape_a = in_shapes[0]
                    shape_b = in_shapes[1]
                    max_len = max(len(shape_a), len(shape_b))
                    shape_a = [1] * (max_len - len(shape_a)) + shape_a
                    shape_b = [1] * (max_len - len(shape_b)) + shape_b
                    out_shape = []
                    for a, b in zip(shape_a, shape_b):
                        if a == b:
                            out_shape.append(a)
                        elif a == 1:
                            out_shape.append(b)
                        elif b == 1:
                            out_shape.append(a)
                        else:
                            out_shape.append(max(a, b))
                elif in_shapes[0]:
                    out_shape = in_shapes[0]
                elif in_shapes[1]:
                    out_shape = in_shapes[1]

            elif op == 'Concat':
                axis = 0
                for attr in node.attribute:
                    if attr.name == 'axis':
                        axis = attr.i
                if all(s is not None for s in in_shapes):
                    out_shape = list(in_shapes[0])
                    total = 0
                    for s in in_shapes:
                        if s[axis] > 0:
                            total += s[axis]
                        else:
                            total = -1
                            break
                    out_shape[axis] = total

            elif op == 'Resize':
                if in_shapes[0]:
                    out_shape = list(in_shapes[0])
                    # 找到 scales 或 sizes
                    scales = None
                    sizes = None
                    # ONNX Resize 输入: (X, roi, scales, sizes)
                    # 2输入: (X, scales) 或 (X, sizes)
                    # 3输入: (X, roi, scales) 或 (X, scales, sizes)
                    # 4输入: (X, roi, scales, sizes)
                    for idx in range(1, len(inputs)):
                        if inputs[idx] in constants:
                            arr = constants[inputs[idx]]
                            if len(arr) == len(out_shape):
                                # 判断是 scales 还是 sizes
                                # 简单方法：如果空间维度的值 > 10，认为是 sizes
                                is_sizes = False
                                for d in range(2, len(out_shape)):
                                    if out_shape[d] > 0 and arr[d] > out_shape[d] and arr[d] > 10:
                                        is_sizes = True
                                        break
                                if is_sizes:
                                    sizes = arr
                                elif scales is None:
                                    scales = arr
                    if scales is not None:
                        for i in range(len(out_shape)):
                            if out_shape[i] > 0:
                                out_shape[i] = int(round(out_shape[i] * float(scales[i])))
                    elif sizes is not None:
                        out_shape = [int(v) for v in sizes]

            elif op == 'Transpose':
                perm = None
                for attr in node.attribute:
                    if attr.name == 'perm':
                        perm = list(attr.ints)
                if in_shapes[0] and perm:
                    out_shape = [in_shapes[0][p] for p in perm]
                else:
                    out_shape = in_shapes[0]

            elif op == 'Reshape':
                if len(inputs) >= 2 and inputs[1] in constants:
                    shape_arr = constants[inputs[1]]
                    out_shape = [int(v) for v in shape_arr.flatten()]
                    # 处理 -1
                    if -1 in out_shape and in_shapes[0]:
                        total = 1
                        for s in in_shapes[0]:
                            total *= s
                        known = 1
                        for s in out_shape:
                            if s != -1:
                                known *= s
                        if known > 0:
                            idx = out_shape.index(-1)
                            out_shape[idx] = total // known

            elif op == 'MatMul':
                if in_shapes[0] and in_shapes[1]:
                    a = in_shapes[0]
                    b = in_shapes[1]
                    if len(a) >= 2 and len(b) >= 2:
                        out_shape = a[:-2] + [a[-2], b[-1]]

            elif op == 'Split':
                axis = 0
                split_sizes = None
                for attr in node.attribute:
                    if attr.name == 'axis':
                        axis = attr.i
                    elif attr.name == 'split':
                        split_sizes = list(attr.ints)
                if split_sizes is None and len(inputs) > 1 and inputs[1] in constants:
                    split_sizes = [int(x) for x in constants[inputs[1]].flatten()]
                if in_shapes[0]:
                    base = list(in_shapes[0])
                    n_out = len(outputs)
                    if split_sizes is None:
                        split_size = base[axis] // n_out if base[axis] > 0 else -1
                        split_sizes = [split_size] * n_out
                    for idx, out in enumerate(outputs):
                        s = list(base)
                        s[axis] = split_sizes[idx] if idx < len(split_sizes) else -1
                        shapes[out] = s
                    continue

            elif op == 'Slice':
                if in_shapes[0]:
                    input_shape = list(in_shapes[0])
                    ndim = len(input_shape)
                    starts = None
                    ends = None
                    axes = list(range(ndim))
                    steps = [1] * ndim
                    # 从输入张量读取
                    if len(inputs) > 1 and inputs[1] in constants:
                        starts = [int(x) for x in constants[inputs[1]].flatten()]
                    if len(inputs) > 2 and inputs[2] in constants:
                        ends = [int(x) for x in constants[inputs[2]].flatten()]
                    if len(inputs) > 3 and inputs[3] in constants:
                        axes = [int(x) for x in constants[inputs[3]].flatten()]
                    if len(inputs) > 4 and inputs[4] in constants:
                        steps = [int(x) for x in constants[inputs[4]].flatten()]
                    if starts is not None and ends is not None:
                        out_shape = list(input_shape)
                        for i, ax in enumerate(axes):
                            if ax < 0:
                                ax += ndim
                            if ax < 0 or ax >= ndim:
                                continue
                            start = starts[i] if i < len(starts) else 0
                            end = ends[i] if i < len(ends) else input_shape[ax]
                            step = steps[i] if i < len(steps) else 1
                            dim = input_shape[ax]
                            if dim <= 0:
                                out_shape[ax] = -1
                                continue
                            if start < 0:
                                start = max(0, start + dim)
                            else:
                                start = min(start, dim)
                            if end < 0:
                                end = max(0, end + dim)
                            else:
                                end = min(end, dim)
                            if step > 0:
                                out_shape[ax] = max(0, (end - start + step - 1) // step)
                            else:
                                out_shape[ax] = max(0, (start - end - step - 1) // (-step))
                    else:
                        out_shape = input_shape

            elif op in ('ReduceMax', 'ReduceSum'):
                axes = None
                keepdims = 1
                for attr in node.attribute:
                    if attr.name == 'axes':
                        axes = list(attr.ints)
                    elif attr.name == 'keepdims':
                        keepdims = attr.i
                if in_shapes[0] and axes:
                    out_shape = list(in_shapes[0])
                    if keepdims:
                        for ax in axes:
                            out_shape[ax] = 1
                    else:
                        for ax in sorted(axes, reverse=True):
                            del out_shape[ax]

            elif op == 'Softmax':
                out_shape = in_shapes[0]

            elif op == 'MaxPool':
                if in_shapes[0] and len(in_shapes[0]) == 4:
                    N, C, H, W = in_shapes[0]
                    kH = kW = 1
                    sH = sW = 1
                    pH = pW = 0
                    for attr in node.attribute:
                        if attr.name == 'kernel_shape':
                            kH, kW = list(attr.ints)
                        elif attr.name == 'strides':
                            sH, sW = list(attr.ints)
                        elif attr.name == 'pads':
                            pH = pW = 0
                            pads = list(attr.ints)
                            if len(pads) >= 2:
                                pH = pads[0]
                                pW = pads[1]
                    outH = (H + 2*pH - kH) // sH + 1
                    outW = (W + 2*pW - kW) // sW + 1
                    out_shape = [N, C, outH, outW]

            elif op == 'Exp':
                out_shape = in_shapes[0]

        except Exception as e:
            pass

        if out_shape:
            for out in outputs:
                shapes[out] = out_shape

    return shapes


# ============================================================
# 5. 主转换函数
# ============================================================
def convert(input_path, output_path):
    print(f"加载模型: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    # 统计原始算子
    orig_ops = defaultdict(int)
    for n in graph.node:
        orig_ops[n.op_type] += 1
    print(f"原始节点数: {len(graph.node)}")

    # 1. SiLU 融合
    graph = fuse_silu(graph)

    # 1.5 Softmax 融合
    graph = fuse_softmax(graph)

    # 1.8 Conv+SiLU 融合
    graph = fuse_conv_silu(graph)

    # 2. 收集常量
    constants = collect_constants(graph)
    print(f"常量张量数: {len(constants)}")

    # 3. 拓扑排序
    sorted_nodes = topo_sort(graph, constants)
    print(f"排序后节点数: {len(sorted_nodes)}")

    # 4. 形状推断
    input_name = graph.input[0].name
    input_shape = [1, 3, 640, 640]
    # 尝试从模型输入获取形状
    for inp in graph.input:
        if inp.type.tensor_type.shape.dim:
            dims = []
            for d in inp.type.tensor_type.shape.dim:
                if d.dim_value:
                    dims.append(d.dim_value)
                elif d.dim_param:
                    dims.append(-1)
                else:
                    dims.append(-1)
            if len(dims) == 4:
                input_shape = dims
                break
    print(f"输入形状: {input_shape}")

    shapes = infer_shapes(sorted_nodes, constants, input_name, input_shape)

    # 5. 分配张量 ID（按照节点处理顺序，保证第一个 Conv 的权重和偏置得到正确的 ID）
    tensor_name_to_id = {}
    tensor_descs = []
    next_id = 0

    def get_tensor_id(name):
        nonlocal next_id
        if name not in tensor_name_to_id:
            tensor_name_to_id[name] = next_id
            next_id += 1
        return tensor_name_to_id[name]

    # 先分配输入
    for inp in graph.input:
        get_tensor_id(inp.name)

    # 按照节点处理顺序分配张量 ID（遇到一个分配一个）
    for node in sorted_nodes:
        for inp in node.input:
            if inp:
                get_tensor_id(inp)
        for out in node.output:
            if out:
                get_tensor_id(out)

    # 分配输出
    output_names = []
    for out in graph.output:
        output_names.append(out.name)
        get_tensor_id(out.name)

    print(f"张量总数: {next_id}")

    # 6. 构建算子描述
    op_descs = []
    for node in sorted_nodes:
        op_type = OP_TYPE_MAP.get(node.op_type)
        if op_type is None:
            print(f"警告: 不支持的算子类型 {node.op_type}，跳过")
            continue

        # 构建属性数据（TLV格式：nlen(1)+name(nlen)+atype(1)+value）
        # atype: 1=FLOAT(4), 2=INT(8), 3=STRING(2+len), 7=INTS(2+cnt*8)
        attr_data = b''

        def make_int_attr(name, value):
            """构建 INT 类型属性 (atype=2, int64)"""
            name_bytes = name.encode('utf-8')
            return struct.pack(f'<B{len(name_bytes)}sBq', len(name_bytes), name_bytes, 2, value)

        def make_ints_attr(name, values):
            """构建 INTS 类型属性 (atype=7, int64数组)"""
            name_bytes = name.encode('utf-8')
            data = struct.pack(f'<B{len(name_bytes)}sBH', len(name_bytes), name_bytes, 7, len(values))
            for v in values:
                data += struct.pack('<q', v)
            return data

        def make_string_attr(name, value):
            """构建 STRING 类型属性 (atype=3)"""
            name_bytes = name.encode('utf-8')
            val_bytes = value if isinstance(value, bytes) else value.encode('utf-8')
            return struct.pack(f'<B{len(name_bytes)}sBH', len(name_bytes), name_bytes, 3, len(val_bytes)) + val_bytes

        if op_type == OP_CONV or op_type == OP_CONVSILU:
            kH = kW = 1
            sH = sW = 1
            pH = pW = 0
            dH = dW = 1
            group = 1
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kH, kW = list(attr.ints)
                elif attr.name == 'strides':
                    sH, sW = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    if len(pads) >= 4:
                        pH = pads[0]
                        pW = pads[1]
                elif attr.name == 'dilations':
                    dH, dW = list(attr.ints)
                elif attr.name == 'group':
                    group = attr.i
            # Conv 直接用 ConvAttr 结构体（9个int32）
            attr_data = struct.pack('<9i', kH, kW, sH, sW, pH, pW, dH, dW, group)

        elif op_type == OP_CONCAT:
            axis = 0
            for attr in node.attribute:
                if attr.name == 'axis':
                    axis = attr.i
            attr_data = make_int_attr('axis', axis)

        elif op_type == OP_TRANSPOSE:
            perm = []
            for attr in node.attribute:
                if attr.name == 'perm':
                    perm = list(attr.ints)
            if perm:
                attr_data = make_ints_attr('perm', perm)

        elif op_type == OP_RESHAPE:
            # 形状从输入张量读取，不需要属性
            pass

        elif op_type == OP_SPLIT:
            axis = 0
            for attr in node.attribute:
                if attr.name == 'axis':
                    axis = attr.i
            attr_data = make_int_attr('axis', axis)

        elif op_type == OP_SLICE:
            # 参数从输入张量读取
            pass

        elif op_type in (OP_REDUCEMAX, OP_REDUCESUM):
            axes = []
            keepdims = 1
            for attr in node.attribute:
                if attr.name == 'axes':
                    axes = list(attr.ints)
                elif attr.name == 'keepdims':
                    keepdims = attr.i
            if axes:
                attr_data += make_ints_attr('axes', axes)
            attr_data += make_int_attr('keepdims', keepdims)

        elif op_type == OP_SOFTMAX:
            axis = -1
            for attr in node.attribute:
                if attr.name == 'axis':
                    axis = attr.i
            attr_data = make_int_attr('axis', axis)

        elif op_type == OP_MAXPOOL:
            kH = kW = 1
            sH = sW = 1
            pH = pW = 0
            for attr in node.attribute:
                if attr.name == 'kernel_shape':
                    kH, kW = list(attr.ints)
                elif attr.name == 'strides':
                    sH, sW = list(attr.ints)
                elif attr.name == 'pads':
                    pads = list(attr.ints)
                    if len(pads) >= 4:
                        pH = pads[0]
                        pW = pads[1]
            attr_data += make_ints_attr('kernel_shape', [kH, kW])
            attr_data += make_ints_attr('strides', [sH, sW])
            attr_data += make_ints_attr('pads', [pH, pW, pH, pW])

        elif op_type == OP_RESIZE:
            mode = b'nearest'
            for attr in node.attribute:
                if attr.name == 'mode':
                    mode = attr.s
            attr_data = make_string_attr('mode', mode)

        # 输入输出 ID
        input_ids = [tensor_name_to_id[inp] for inp in node.input if inp]
        output_ids = [tensor_name_to_id[out] for out in node.output if out]

        op_descs.append({
            'op_type': op_type,
            'attr_data': attr_data,
            'input_ids': input_ids,
            'output_ids': output_ids,
        })

    print(f"算子总数: {len(op_descs)}")

    # 7. 序列化常量权重数据
    weight_data = b''
    tensor_offsets = {}

    # 按张量 ID 顺序排列常量
    for name, tid in sorted(tensor_name_to_id.items(), key=lambda x: x[1]):
        if name in constants:
            arr = constants[name]
            tensor_offsets[tid] = len(weight_data)
            if arr.dtype == np.float32:
                weight_data += arr.astype(np.float32).tobytes()
            elif arr.dtype == np.int64:
                weight_data += arr.astype(np.int64).tobytes()
            elif arr.dtype == np.int32:
                weight_data += arr.astype(np.int32).tobytes()
            else:
                weight_data += arr.astype(np.float32).tobytes()

    print(f"权重数据大小: {len(weight_data)} bytes")

    # 8. 写入文件
    with open(output_path, 'wb') as f:
        # 文件头 (20 bytes)
        header = struct.pack('<IBBBHHII',
            0x314F5954,  # magic "TYO1"
            1,             # version
            1,             # input_count
            len(output_names),  # output_count
            len(op_descs),      # op_count
            next_id,            # tensor_count
            len(weight_data),   # weights_size
            0                   # reserved
        )
        f.write(header)

        # 输入名称
        input_name_str = input_name.encode('utf-8')
        f.write(struct.pack('<H', len(input_name_str)))
        f.write(input_name_str)

        # 输出名称
        for out_name in output_names:
            out_str = out_name.encode('utf-8')
            f.write(struct.pack('<H', len(out_str)))
            f.write(out_str)

        # 张量描述
        for name, tid in sorted(tensor_name_to_id.items(), key=lambda x: x[1]):
            # 常量张量的形状从 constants 读取，中间张量的形状从 shapes 读取
            if name in constants:
                shape = list(constants[name].shape)
            else:
                shape = shapes.get(name, [])
            ndim = len(shape)
            is_const = 1 if name in constants else 0
            dtype = 0  # float32 默认
            if name in constants:
                arr = constants[name]
                if arr.dtype == np.int64:
                    dtype = 1
                elif arr.dtype == np.int32:
                    dtype = 2
            offset = tensor_offsets.get(tid, 0)

            # TensorDesc: 12 bytes + ndim*4 bytes
            desc = struct.pack('<HBBBBBBI',
                tid, ndim, dtype, is_const, 0, 0, 0, offset
            )
            f.write(desc)
            for s in shape:
                f.write(struct.pack('<i', s))

        # 算子描述
        for op in op_descs:
            # OpDesc: 4 bytes header + attr + input_ids + output_ids
            header = struct.pack('<BBBB',
                op['op_type'],
                len(op['input_ids']),
                len(op['output_ids']),
                len(op['attr_data'])
            )
            f.write(header)
            f.write(op['attr_data'])
            for iid in op['input_ids']:
                f.write(struct.pack('<H', iid))
            for oid in op['output_ids']:
                f.write(struct.pack('<H', oid))

        # 权重数据
        f.write(weight_data)

    print(f"\n转换完成: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path)} bytes")


if __name__ == '__main__':
    import os
    if len(sys.argv) < 3:
        print("用法: python convert_model.py <input.onnx> <output.tyro>")
        sys.exit(1)
    convert(sys.argv[1], sys.argv[2])
