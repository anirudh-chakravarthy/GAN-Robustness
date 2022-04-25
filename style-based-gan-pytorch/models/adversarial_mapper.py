import torch
import torch.nn as nn

class ResBlock(nn.Module):

    def __init__(self, in_features=512, out_features=512):
        super(ResBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(in_features, out_features, bias=True),
            nn.ReLU(),
            nn.Linear(out_features, out_features, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        out = out + x
        return out


class Generator(nn.Module):
    def __init__(self, in_features=512, out_feature=512, num_res_blocks=3):
        super(Generator, self).__init__()

        self.dense = nn.Linear(in_features=in_features, out_features=out_feature, bias=True)
        self.layers = [ResBlock(in_features=in_features) for _ in range(num_res_blocks)]
        self.final = [
            nn.BatchNorm1d(num_features=in_features),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512)
        ]
        self.layers.extend(self.final)
        self.layers = nn.ModuleList([*self.layers])


    def forward(self, x):
        out = self.dense(x)
        for layer in self.layers:
            out = layer(out)
        return out


# ----------- UNIT TESTS ------------- #

def test_resblock():
    x = torch.randn(size=(10, 512))
    resblock = ResBlock(in_features=512, out_features=512)
    print(resblock(x).shape)

def test_generator():
    x = torch.randn(size=(10, 512))
    gen = Generator()
    print(gen)
    print(gen(x).shape)


if __name__ == "__main__":
    test_generator()