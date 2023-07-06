import torch.nn as nn

class FCnet(nn.Module):
    def __init__(self):
        super(FCnet,self).__init__()
        self.line1 = nn.Linear(100, 1)
        self.rl = nn.Sigmoid()

    def forward(self,x):
        x = self.line1(x)
        x = self.rl(x)
        return x

