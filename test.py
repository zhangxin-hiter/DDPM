import torch.nn as nn
import torch

emb = nn.Embedding(10, 10)
print(emb(torch.randint(0, 10, (3, )).long()).shape)