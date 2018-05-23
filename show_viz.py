from models import *
import torch
from viz import make_dot, make_dot_from_trace

input_ = torch.randn((1, 3, 32, 32)).cuda()

model = HiResC(BasicBlock, [3, 3, 3])
model = model.cuda()

y = model(input_)
make_dot(y.mean(), params=dict(model.named_parameters())).view()
