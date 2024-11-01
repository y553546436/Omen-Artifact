import torch
import torch.nn as nn
import math
from tqdm import tqdm

hv_dtype = torch.int8

class BinaryLevelEncoder:
    def __init__(self, features, dim, levels, min_val, max_val):
        torch.manual_seed(0)
        self.features = features
        self.dim = dim
        self.num_level = levels
        self.min_val = min_val
        self.max_val = max_val

        self.basis = torch.randint(0, 2, (self.features, self.dim), dtype=hv_dtype)
        self.codebook = torch.empty(levels + 2, self.dim, dtype=hv_dtype)
        xor_pos = torch.randperm(self.dim)
        now = torch.randint(0, 2, (self.dim,), dtype=hv_dtype)
        self.codebook[0] = now.clone()
        step_len = math.ceil(self.dim / levels + 2)
        ones = torch.ones(self.dim, dtype=hv_dtype)
        self.thresholds = torch.linspace(min_val, max_val, levels + 1)
        for level, index in enumerate(range(0, self.dim, step_len)):
            self.codebook[level + 1] = now.clone()
            now[xor_pos[index:min(index + step_len, self.dim)]] ^= ones[index:min(index + step_len, self.dim)]
        self.codebook[-1] = now.clone()


    def to(self, device):
        self.basis = self.basis.to(device)
        self.codebook = self.codebook.to(device)
        self.thresholds = self.thresholds.to(device)
        return self


    def __call__(self, x, batch_size=100):
        return self.encode(x, batch_size=batch_size)


    def map_to_levels(self, values):
        levels = torch.zeros_like(values, dtype=torch.int32)
        for threshold in self.thresholds:
            levels += values > threshold
        return levels


    def encode(self, x, batch_size=100):
        hvs = torch.empty(x.size(0), self.dim, dtype=hv_dtype, device=x.device)
        with tqdm(total=x.size(0), desc='Encoding') as pbar:
            for i in range(0, x.size(0), batch_size):
                x_batch = x[i:i + batch_size]
                batch_levels = self.map_to_levels(x_batch)
                batch_mapped = self.codebook[batch_levels]
                batch_sum = torch.bitwise_xor(batch_mapped, self.basis).mean(dim=-2, dtype=torch.float)
                batch_hvs = torch.empty_like(batch_sum, dtype=hv_dtype, device=x_batch.device)
                batch_hvs[batch_sum > 0.5] = 1
                batch_hvs[batch_sum <= 0.5] = 0
                batch_hvs[batch_sum == 0.5] = torch.randint(0, 2, batch_hvs.shape, dtype=hv_dtype, device=x_batch.device)[batch_sum == 0.5]
                hvs[i:i + batch_size] = batch_hvs
                pbar.update(batch_size)
        return hvs


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = BinaryLevelEncoder(3, 10000, 8, 0, 1)
    print(enc.codebook)
    print(enc.thresholds)
    print(enc.codebook.shape)

    random_tests = torch.rand(4, 3)
    encoded_tests = enc.encode(random_tests)
    print(encoded_tests.shape)
    print(encoded_tests)
    for level in enc.codebook:
        print(level.float().mean())
