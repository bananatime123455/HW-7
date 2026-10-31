import torch
from student_code import LeNet, count_model_params

m = LeNet()
out, shapes = m(torch.randn(2, 3, 32, 32))
print(shapes)
print(count_model_params())