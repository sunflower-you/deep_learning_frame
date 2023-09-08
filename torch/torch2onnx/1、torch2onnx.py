import torch
from torch.nn import Module
import torchvision

# 加载预训练的PyTorch模型
model = torchvision.models.resnet18(pretrained=True)

# 将模型设置为评估模式
model.eval()

# 创建一个虚拟的输入张量
dummy_input = torch.randn(1, 3, 224, 224)

# 导出模型为ONNX格式
torch.onnx.export(model, dummy_input, "model.onnx")
