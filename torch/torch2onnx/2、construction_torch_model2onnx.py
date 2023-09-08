"""
构造单个op的argmax的torch模型，并转换成onnx
"""

import torch
from torch.nn import Module
import torchvision

'''
    case1: 
        torch  argmax 的dim默认是None，即 输入tensor展平之后，最大元素索引为输出
            此时torch转onnx时候，先用Reshape展平，再用argmax计算
'''
class My_Model1(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.argmax(x)
    

# 加载预训练的PyTorch模型
model = My_Model1()

# 将模型设置为评估模式
model.eval().cpu()

# 创建一个虚拟的输入张量
dummy_input = torch.arange(0,9)
# dummy_input = torch.arange(0,9).reshape(3,3)

# 导出模型为ONNX格式
torch.onnx.export(model, dummy_input, "argmax_none_dim.onnx")

output = model(dummy_input)
print(output)  # tensor(8)

'''
    case2: 
        torch  argmax 的dim是-1，输入是1维的
'''
class My_Model2(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return torch.argmax(x,dim=-1)
    

# 加载预训练的PyTorch模型
model = My_Model2()

# 将模型设置为评估模式
model.eval().cpu()

# 创建一个虚拟的输入张量
dummy_input = torch.arange(0,9)
# dummy_input = torch.arange(0,9).reshape(3,3)

# 导出模型为ONNX格式
torch.onnx.export(model, dummy_input, "argmax_dim-1.onnx")

output = model(dummy_input)
print(output)  # tensor(8)

'''
    case3: 
        torch  argmax 的dim是-1，输入是2维的
'''
dummy_input = torch.arange(0,9).reshape(3,3)
# 导出模型为ONNX格式
torch.onnx.export(model, dummy_input, "argmax_dim0.onnx")

output = model(dummy_input)
print(dummy_input)
'''
tensor([[0, 1, 2],
        [3, 4, 5],
        [6, 7, 8]])

'''
print(output)  # tensor([2, 2, 2])
