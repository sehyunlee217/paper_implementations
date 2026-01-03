import torch
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self, num_channels, train, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

        self.train = train
        self.momentum = momentum
        self.register_buffer("running_mean", torch.zeros(1, num_channels, 1))
        self.register_buffer("running_variance", torch.ones(1, num_channels, 1))

    def forward(self, x):
        # batchnorm across the number of features (channels) -> want to flatten other dims
        # x: (Batch, Channel, Width, Height) => (Batch, Channel, Feature_Size)
        init_shape = x.shape
        x = x.flatten(2)

        if self.train:
            # take mean across features
            # mean, mean_sq : (Channel) -> keepdim=True -> (1, Channel, 1)
            mean = x.mean(dim=(0, 2), keepdim=True)
            mean_sq = (x**2).mean(dim=(0, 2), keepdim=True)

            # Var = E[x**2] - E[x]**2
            var = mean_sq - mean**2

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_variance = (
                1 - self.momentum
            ) * self.running_variance + self.momentum * var

            mean = mean
            var = var
        else:
            mean = self.running_mean
            var = self.running_variance

        x_norm = (x - mean) / (torch.sqrt(var + self.eps))

        output = self.gamma * x_norm + self.beta

        return output.view(init_shape)


sample_batch = torch.rand(32, 3, 64, 64)
bn_layer = BatchNorm(num_channels=3, train=True)

out = bn_layer.forward(sample_batch)
print(f"dim(out): {out.shape}")
