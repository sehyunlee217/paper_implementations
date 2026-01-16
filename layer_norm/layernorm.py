import matplotlib.pyplot as plt
import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        batch_size, length, feature_dim = x.shape
        output = torch.zeros_like(x)

        for i in range(batch_size):
            for t in range(length):
                word_vector = x[i, t, :]

                mu = word_vector.sum() / feature_dim

                var = ((word_vector - mu) ** 2).sum() / feature_dim

                output[i, t, :] = (word_vector - mu) / torch.sqrt(var + self.eps)

        return output


class LayerNorm2(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        batch_size, length, feature_dim = x.shape
        output = torch.zeros_like(x)

        # dim=-1 so we squish along last idx(feature_dim)
        # keep_dim=True so we get (batch_size, length, 1)
        mu = x.mean(dim=-1, keep_dim=True)
        var = x.var(dim=-1, keep_dim=True)

        output = (x - mu) / torch.sqrt(var + self.eps)

        return output


# Assume feature vecs from a batch 32, sentence of length 64, and feature dim. of 512.
# We want to normalize across each word in a batch.
sample_batch = torch.randn(32, 64, 512) * 5 + 10

print(sample_batch[0, 0, :])
ln_layer = LayerNorm()
out = ln_layer.forward(sample_batch)
print(f"dim(out): {out.shape}")
print(out[0, 0, :])


# -- this is made by Gemini --
def plot_distributions(before, after):
    # Flatten the tensors to look at all values across the batch/time/features
    before_flat = before.flatten().detach().cpu().numpy()
    after_flat = after.flatten().detach().cpu().numpy()

    plt.figure(figsize=(10, 6))

    # Plot 'Before' distribution
    plt.hist(
        before_flat, bins=100, alpha=0.5, label="Before LayerNorm (Raw)", color="blue"
    )

    # Plot 'After' distribution
    plt.hist(
        after_flat,
        bins=100,
        alpha=0.5,
        label="After LayerNorm (Normalized)",
        color="orange",
    )

    plt.axvline(x=0, color="red", linestyle="--", alpha=0.7, label="Mean = 0")
    plt.title("Effect of Layer Normalization on Value Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.show()


# Usage:
plot_distributions(sample_batch, out)
