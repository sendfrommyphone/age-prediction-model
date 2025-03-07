import torch
import torch.onnx
from train import AgePredictor  # 导入你的模型类

# 加载模型权重
model = AgePredictor()  # 初始化模型实例（需与训练时的结构一致）
model.load_state_dict(torch.load("best_con_tiny_model.pth"))
model.eval()  # 切换到推理模式

# 生成虚拟输入（示例）
dummy_input = torch.randn(1, 3, 224, 224)  # 输入形状需与训练时一致（例如：batch, channels, height, width）

# 导出为 ONNX
onnx_path = "model.onnx"
torch.onnx.export(
    model,                     # 模型实例
    dummy_input,               # 虚拟输入数据
    onnx_path,                 # 输出路径
    export_params=True,        # 保存模型权重
    opset_version=17,          # ONNX 算子版本（推荐 11+，根据需求调整）
    do_constant_folding=True,  # 优化常量折叠
    input_names=["input"],     # 输入名称（自定义，用于推理时识别）
    output_names=["output"],   # 输出名称
    dynamic_axes={             # 动态维度（可选，例如支持可变 batch_size）
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print(f"ONNX 模型已保存到: {onnx_path}")