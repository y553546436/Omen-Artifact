import math
import torch

class Encoder:
    '''
    The nonlinear encoder class maps data nonlinearly to high dimensional space.
    To do this task, it uses two randomly generated tensors:

    :math:`B`. The `(dim, features)` sized random basis hypervectors, drawn
    from a standard normal distribution
    :math:`b`. An additional `(dim,)` sized base, drawn from a uniform
    distribution between :math:`[0, 2\pi]`.

    The hypervector :math:`H \in \mathbb{R}^D` of :math:`X \in \mathbb{R}^f`
    is:

    .. math:: H_i = \cos(X \cdot B_i + b_i) \sin(X \cdot B_i)

    Args:
        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): Target dimension for output data.
    '''
    def __init__(self, features : int, dim : int = 4000):
        self.dim = dim
        self.features = features
        self.basis = torch.randn(self.dim, self.features)
        self.base = torch.empty(self.dim).uniform_(0.0, 2*math.pi)

    def to(self, device):
        self.basis = self.basis.to(device)
        self.base = self.base.to(device)
        return self

    def encode(self, x : torch.Tensor):
        '''
        Encodes each data point in `x` to high dimensional space.
        The encoded representation of the `(n?, features)` samples described
        in :math:`x`, is the `(n?, dim)` matrix :math:`H`:

        .. math:: H_{ij} = \cos(x_i \cdot B_j + b_j) \sin(x_i \cdot B_j)

        Note:
            This encoder is very sensitive to data preprocessing. Try
            making input have unit norm (normalizing) or standarizing each
            feature to have mean=0 and std=1/sqrt(features) (scaling).

        Args:
            x (:class:`torch.Tensor`): The original data points to encode. Must
                have size `(n?, features)`.

        Returns:
            :class:`torch.Tensor`: The high dimensional representation of each
            of the `n?` data points in x, which respects the equation given
            above. It has size `(n?, dim)`.
        '''

        n = x.size(0)
        bsize = math.ceil(0.01*n)
        h = torch.empty(n, self.dim, device=x.device, dtype=x.dtype)

        # we need batches to control memory usage
        for i in range(0, n, bsize):
            temp = torch.matmul(x[i:i+bsize], self.basis.T)
            h[i:i+bsize] = temp + self.base
            h[i:i+bsize].cos_().mul_(temp.sin_())
        return h
    

    def __call__(self, x: torch.Tensor):
        return self.encode(x)


    def basis_orthogonality(self):
        '''
        Returns the orthogonality of the basis vectors. Which is the cosine similarity of the basis vectors.

        Returns:
            :class:`torch.Tensor`: The orthogonality of the basis vectors.
        '''
        dot = torch.mm(self.basis, self.basis.T).abs()
        norm = torch.norm(self.basis, dim=1).unsqueeze(1)
        sim_mat = dot / (norm * norm.T)
        sim_mat.fill_diagonal_(0)
        # delete the diagonal elements
        return sim_mat
