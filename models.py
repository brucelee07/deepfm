import torch.nn as nn
import torch
import torch.nn.functional as F


class Fm(nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, input_x):
        square_of_sum = torch.sum(input_x, dim=1)**2
        sum_of_square = torch.sum(input_x**2, dim=1)
        output = square_of_sum - sum_of_square
        if self.reduce_sum:
            output = torch.sum(output, dim=1, keepdim=True)
        output = 0.5 * output
        return output


class Dense(nn.Module):

    def __init__(self, hidden_units, dropout=0.2):
        super().__init__()
        self.dens = nn.ModuleList([
            nn.Linear(unit[0], unit[1])
            for unit in zip(hidden_units[:-1], hidden_units[1:])
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for linear in self.dens:

            x = linear(x)
            x = F.relu(x)

        return self.dropout(x)


class Deep_Fm(nn.Module):

    def __init__(
        self,
        features,
        hidden_units,
        embedding_dim=10,
    ):
        super().__init__()
        self.dens_feature, self.sparse_feature = features

        self.embedding_layers = nn.ModuleDict({
            f"embed_{i}": nn.Embedding(self.sparse_feature[feat],
                                       embedding_dim)
            for i, feat in enumerate(self.sparse_feature)
        })

        hidden_units.insert(
            0,
            len(self.dens_feature) + len(self.sparse_feature) * embedding_dim,
        )

        self.dens = Dense(hidden_units)
        self.fm = Fm(reduce_sum=False)
        self.out = nn.Linear(2, 1)

    def forward(self, x):
        dense_input, sparse_input = (
            x[:, :len(self.dens_feature)],
            x[:, len(self.dens_feature):],
        )

        sparse_input = sparse_input.long()
        sparse_embeds = [
            self.embedding_layers[f"embed_{i}"](sparse_input[:, i])
            for i in range(sparse_input.shape[1])
        ]
        sparse_embeds_ = torch.cat(sparse_embeds, dim=-1)
        x = torch.cat((sparse_embeds_, dense_input), dim=-1)

        deep_out = self.dens(x)
        fm_out = self.fm(x)
        fm_out = fm_out.unsqueeze(-1)
        x = torch.cat((fm_out, deep_out), dim=-1)
        x = torch.sigmoid(self.out(x))
        return x
