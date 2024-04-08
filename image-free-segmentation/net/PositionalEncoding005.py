import torch
import torch.nn as nn


#实现了位置编码
class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_length=512):
        super(FixedPositionalEncoding, self).__init__()

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return x


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()


        #self.position_embeddings = nn.Parameter(torch.zeros(1,64,480)) #0.15
        #self.position_embeddings = nn.Parameter(torch.zeros(1,64,320)) #0.1
        #self.position_embeddings = nn.Parameter(torch.zeros(1,64,224)) #0.07
        self.position_embeddings = nn.Parameter(torch.zeros(1,64,160)) #0.05
        #self.position_embeddings = nn.Parameter(torch.zeros(1,64,96))  #0.03

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        return x + position_embeddings
