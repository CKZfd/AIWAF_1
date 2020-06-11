import torch.nn as nn
import torch
import torch.nn.functional as F

class aiwafNet(nn.Module):
    def __init__(self, seq_len, embed_dim, hidden, num_class):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, embed_dim)
        # sparse # bool值，设置成True时参数weight为稀疏tensor。
        # 所谓稀疏tensor是说反向传播时只更新当前使用词的embedding，加快更新速度。
        self.fc1 = nn.Linear(embed_dim*seq_len, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], num_class)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.5
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc1.weight.data.uniform_(-initrange, initrange)
    #     self.fc1.bias.data.zero_()
    #     self.fc2.weight.data.uniform_(-initrange, initrange)
    #     self.fc2.bias.data.zero_()
    #     self.fc3.weight.data.uniform_(-initrange, initrange)
    #     self.fc3.bias.data.zero_()

    def forward(self, x):
        x = x.long()
        embedded = self.embedding(x)
        embedded = embedded.view(embedded.size(0), -1)
        out = F.relu(self.fc1(embedded))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out