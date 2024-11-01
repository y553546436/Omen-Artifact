import torch
import math
from tqdm import tqdm
import numpy as np

hv_dtype = torch.int8

class SentenceEncoder:
    def __init__(self, dim, chars):
        torch.manual_seed(0)
        np.random.seed(0)
        self.dim = dim
        assert dim % 64 == 0, 'dim must be a multiple of 64'
        # create a mapping from characters to indices
        self.char_map = {}
        for i, char in enumerate(sorted(chars)):
            self.char_map[char] = i
        self.codebook = torch.randint(0, 2, (len(chars), dim), dtype=hv_dtype)


    def to(self, device):
        self.codebook = self.codebook.to(device)
        return self


    def __call__(self, x, batch_size=100):
        return self.encode(x, batch_size=batch_size)


    def get_trigram_hv(self, trigram):
        hv = torch.zeros(self.dim, dtype=hv_dtype, device=self.codebook.device)
        for i, char in enumerate(trigram):
            # bitwise xor of code cyclic permuted by i
            hv ^= self.codebook[self.char_map[char]].roll(-i*64, dims=0)
        return hv


    def get_trigram_hvs(self, x_batch):
        # x_batch is a list of strings with varying lengths
        # returns a tensor of shape (batch_size, # of trigrams, dim)
        lengths = torch.tensor([len(s)-2 for s in x_batch], device=self.codebook.device)
        trigram_hvs = torch.zeros(len(x_batch), torch.max(lengths), self.dim, dtype=hv_dtype, device=self.codebook.device)
        for i, s in enumerate(x_batch):
            for j in range(len(s) - 2):
                trigram = s[j:j + 3]
                trigram_hvs[i, j] = self.get_trigram_hv(trigram)
        return trigram_hvs, lengths


    def encode(self, x, batch_size=100):
        # input x is a list of strings
        hvs = torch.empty(len(x), self.dim, dtype=hv_dtype, device=self.codebook.device)
        with tqdm(total=len(x), desc='Encoding') as pbar:
            for i in range(0, len(x), batch_size):
                x_batch = x[i:i + batch_size]
                batch_trigram_hvs, lengths = self.get_trigram_hvs(x_batch) # shape (batch_size, # of trigrams, dim)
                # batch_sum = torch.bitwise_xor(batch_mapped, self.basis).mean(dim=-2, dtype=torch.float)
                batch_sum = batch_trigram_hvs.sum(dim=-2, dtype=torch.float)
                batch_hvs = torch.empty_like(batch_sum, dtype=hv_dtype, device=self.codebook.device)
                # broadcast lengths to match batch_sum shape
                lengths = lengths.unsqueeze(-1).expand_as(batch_sum)
                batch_hvs[batch_sum > lengths/2] = 1
                batch_hvs[batch_sum < lengths/2] = 0
                batch_hvs[batch_sum == lengths/2] = torch.randint(0, 2, batch_hvs.shape, dtype=hv_dtype, device=self.codebook.device)[batch_sum == lengths/2]
                hvs[i:i + batch_size] = batch_hvs
                pbar.update(batch_size)
        return hvs


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = SentenceEncoder(5, set('abc'))
    print(enc.codebook)

    # test on random 5-letter words in 'abc'
    # x = ['abc', 'cba', 'bca', 'cab']
    x = ['cba', 'baa', 'aab', 'abc', 'bca', 'cab', 'abb', 'bbb', 'cbaabcabbb']
    hvs = enc(x)
    print(hvs)
