import math
from typing import Union
import torch

from onlinehd_encoder import Encoder


def cos_cdist(x1 : torch.Tensor, x2 : torch.Tensor, eps : float = 1e-8):
    r'''
    Computes pairwise cosine similarity between samples in `x1` and `x2`,
    forcing each point l2-norm to be at least `eps`. This similarity between
    `(n?, f?)` samples described in :math:`x1` and the `(m?, f?)` samples
    described in :math:`x2` with scalar :math:`\varepsilon > 0` is the
    `(n?, m?)` matrix :math:`\delta` given by:

    .. math:: \delta_{ij} = \frac{x1_i \cdot x2_j}{\max\{\|x1_i\|, \varepsilon\} \max\{\|x2_j\|, \varepsilon\}}

    Args:
        x1 (:class:`torch.Tensor`): The `(n?, f?)` sized matrix of datapoints
            to score with `x2`.

        x2 (:class:`torch.Tensor`): The `(m?, f?)` sized matrix of datapoints
            to score with `x1`.

        eps (float, > 0): Scalar to prevent zero-norm vectors.

    Returns:
        :class:`torch.Tensor`: The `(n?, m?)` sized tensor `dist` where
        `dist[i,j] = cos(x1[i], x2[j])` given by the equation above.

    '''
    eps = torch.tensor(eps, device=x1.device)
    norms1 = x1.norm(dim=1).unsqueeze_(1).max(eps)
    norms2 = x2.norm(dim=1).unsqueeze_(0).max(eps)
    cdist = x1 @ x2.T
    cdist.div_(norms1).div_(norms2)
    return cdist


class OnlineHD(object):
    '''
    Hyperdimensional classification algorithm. OnlineHD utilizes a `(c, d)`
    sized tensor for the model initialized with zeros. Every `d`-sized vector on
    this matrix will be the high dimensional representation of each class,
    called class hypervector.

    Args:
        classes (int, > 0): The number of classes of the problem.

        features (int, > 0): Dimensionality of original data.

        dim (int, > 0): The target dimensionality of the high dimensional
            representation.

    Example:
        >>> import onlinehd
        >>> dim = 10000
        >>> n_samples = 1000
        >>> features = 100
        >>> clusters = 5
        >>> x = torch.randn(n_samples, features) # dummy data
        >>> y = torch.randint(0, classes, [n_samples]) # dummy data
        >>> model = onlinehd.OnlineHD(classes, features, dim=dim)
        >>> if torch.cuda.is_available():
        ...     print('Training on GPU!')
        ...     model = model.to('cuda')
        ...     x = x.to('cuda')
        ...     y = y.to('cuda')
        ...
        Training on GPU!
        >>> model.fit(x, y, epochs=10)
        >>> ypred = model(x)
        >>> ypred.size()
        torch.Size([1000])
    '''
    def __init__(self, classes : int, features : int, dim : int = 4000):
        self.classes = classes
        self.dim = dim
        self.encoder = Encoder(features, dim)
        self.model = torch.zeros(self.classes, self.dim)

    def __call__(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the predicted class of each data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to predict. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The predicted class of each data point.
            Has size `(n?,)`.
        '''

        return self.scores(x, encoded=encoded).argmax(1)

    def predict(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the predicted cluster of each data point in x. See
        :func:`__call__` for details.
        '''

        return self(x, encoded=encoded)

    def probabilities(self, x : torch.Tensor, encoded : bool = False):
        '''
        Returns the probabilities of belonging to a certain class for each
        data point in x.

        Args:
            x (:class:`torch.Tensor`): The data points to use. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The class probability of each data point.
            Has size `(n?, classes)`.
        '''


        return self.scores(x, encoded=encoded).softmax(1)

    def scores(self, x : torch.Tensor, encoded : bool = False):
        r'''
        Returns pairwise cosine similarity between datapoints in `x` and
        each class hypervector. Calling `model.scores(x, encoded=True)` is
        the same as `spatial.cos_cdist(x, model.model)`.

        Args:
            x (:class:`torch.Tensor`): The data points to score. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

        Returns:
            :class:`torch.Tensor`: The cosine similarity between encoded input
            data and class hypervectors.

        See Also:
            :func:`spatial.cos_cdist` for details.
        '''

        h = x if encoded else self.encode(x)
        return cos_cdist(h, self.model)

    def encode(self, x : torch.Tensor):
        '''
        Encodes input data

        See Also:
            :class:`onlinehd.Encoder` for more information.
        '''

        return self.encoder(x)

    def fit(self,
            x : torch.Tensor,
            y : torch.Tensor,
            encoded : bool = False,
            lr : float = 0.035,
            epochs : int = 120,
            batch_size : Union[int, None, float] = 1024,
            bootstrap : Union[float, str] = 0.01):
        '''
        Starts learning process using datapoints `x` as input points and `y`
        as their labels.

        Args:
            x (:class:`torch.Tensor`): Input data points. Must
                have size `(n?, dim)` if `encoded=False`, otherwise must
                have size `(n?, features)`.

            encoded (bool): Specifies if input data is already encoded.

            lr (float, > 0): Learning rate.

            epochs (int, > 0): Max number of epochs allowed.

            batch_size (int, > 0 and <= n?, or float, > 0 and <= 1, or None):
                If int, the number of samples to use in each batch. If float,
                the fraction of the samples to use in each batch. If none the
                whole dataset will be used per epoch (same if used 1.0 or n?).

            one_pass_fit (bool): Whether to use onepass learning process or not.
                If true, iterative method will be used after one pass fit
                anyways for the number of epochs specified.

            bootstrap (float, > 0, <= 1 or 'single-per-class'): In order to
                initialize class hypervectors, OnlineHD does naive accumulation
                with a small fragment of data. This portion is determined by
                this argument. If 'single-per-class' is used, a single datapoint
                per class will be used as starting class hypervector.

        Warning:
            Using `one_pass_fit` is not advisable for very large data or
            while using GPU. It is expected to see high memory usage using
            this option and it does not benefit from paralellization.

        Returns:
            :class:`OnlineHD`: self
        '''

        h = x if encoded else self.encode(x)
        self._iterative_fit(h, y, lr, epochs, batch_size)
        return self

    def to(self, *args):
        '''
        Moves data to the device specified, e.g. cuda, cpu or changes
        dtype of the data representation, e.g. half or double.
        Because the internal data is saved as torch.tensor, the parameter
        can be anything that torch accepts. The change is done in-place.

        Args:
            device (str or :class:`torch.torch.device`) Device to move data.

        Returns:
            :class:`OnlineHD`: self
        '''


        self.model = self.model.to(*args)
        self.encoder = self.encoder.to(*args)
        return self

    def _iterative_fit(self, h, y, lr, epochs, batch_size):
        n = h.size(0)
        for epoch in range(epochs):
            for i in range(0, n, batch_size):
                h_ = h[i:i+batch_size]
                y_ = y[i:i+batch_size]
                scores = self.scores(h_, encoded=True)
                y_pred = scores.argmax(1)
                wrong = y_ != y_pred

                # computes alphas to update model
                # alpha1 = 1 - delta[lbl] -- the true label coefs
                # alpha2 = delta[max] - 1 -- the prediction coefs
                aranged = torch.arange(h_.size(0), device=h_.device)
                alpha1 = (1.0 - scores[aranged,y_]).unsqueeze_(1)
                alpha2 = (scores[aranged,y_pred] - 1.0).unsqueeze_(1)

                for lbl in y_.unique():
                    m1 = wrong & (y_ == lbl) # mask of missed true lbl
                    m2 = wrong & (y_pred == lbl) # mask of wrong preds
                    self.model[lbl] += lr*(alpha1[m1]*h_[m1]).sum(0)
                    self.model[lbl] += lr*(alpha2[m2]*h_[m2]).sum(0)
