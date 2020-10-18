import torch.nn as nn
import torch
import numpy as np
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
print(input)
target = torch.empty(3).random_(2)
print(target)
output = loss(m(input), target)
print(output)
output.backward()
print(output)
