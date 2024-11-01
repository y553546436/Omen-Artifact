import torch

class BinaryOnlineHD:
    def __init__(self, classes, dim=10000, encoder=None):
        torch.manual_seed(0)
        self.classes = classes
        self.dim = dim
        self.encoder = encoder
        self.model = torch.zeros(self.classes, self.dim, dtype=torch.float)
        self.total = torch.zeros(self.classes, dtype=torch.float)

    def encode(self, x, batch_size=100):
        return self.encoder(x, batch_size=batch_size)


    def to(self, device):
        self.model = self.model.to(device)
        self.total = self.total.to(device)
        self.encoder = self.encoder.to(device)
        return self


    def __call__(self, x: torch.Tensor):
        # Assume the input is encoded
        return self.predict(x)


    def predict(self, x: torch.Tensor):
        # Assume the input is encoded
        assert x.shape[-1] == self.dim, "input may not be encoded"
        diss = self.diss(x)
        return diss.argmin(dim=-1)


    def quantized_model(self):
        quantized_model = torch.empty_like(self.model, dtype=torch.int8, device=self.model.device)
        threshold = (self.total / 2).unsqueeze(-1)
        quantized_model[self.model > threshold] = 1
        quantized_model[self.model <= threshold] = 0
        quantized_model[self.model == threshold] = torch.randint(0, 2, self.model.shape, dtype=torch.int8, device=self.model.device)[self.model == threshold]
        return quantized_model


    def diss(self, x: torch.Tensor):
        quantized_model = self.quantized_model()
        return (x.unsqueeze(-2) ^ quantized_model).mean(dim=-1, dtype=torch.float)


    def first_pass(self, x: torch.Tensor, y: torch.Tensor, lr: float = 1.0):
        for label in range(self.classes):
            mask = y == label
            self.model[label] += lr * x[mask].sum(dim=0)
            self.total[label] += lr * mask.sum().float()
        return self


    def fit(self, x: torch.Tensor, y: torch.Tensor, x_test: torch.Tensor, y_test: torch.Tensor, epochs=100, batch_size: int=1, lr: float = 1.0, decay=0.98):
        self.first_pass(x, y)
        for epoch in range(epochs):
            for i in range(0, x.size(0), batch_size):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                diss = self.diss(x_batch)
                pred = diss.argmin(dim=-1)
                correct_class_diss = diss[torch.arange(x_batch.size(0)), y_batch]
                pred_diss = diss[torch.arange(x_batch.size(0)), pred]
                mispred_mask = pred != y_batch
                correct_class_update = lr * (1 - correct_class_diss)
                pred_update = - lr * (1 - pred_diss)
                self.model[pred[mispred_mask]] += pred_update[mispred_mask].unsqueeze(-1) * x_batch[mispred_mask]
                self.model[y_batch[mispred_mask]] += correct_class_update[mispred_mask].unsqueeze(-1) * x_batch[mispred_mask]
                self.total[pred[mispred_mask]] += pred_update[mispred_mask]
                self.total[y_batch[mispred_mask]] += correct_class_update[mispred_mask]
            lr *= decay

            # Test
            diss = self.diss(x_test)
            pred = diss.argmin(dim=-1)
            acc = (pred == y_test).float().mean().item()
            print(f'Epoch {epoch + 1}/{epochs} - Test Accuracy: {acc}')
        return self




if __name__ == '__main__':
    def diss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.unsqueeze(-2) ^ b).mean(dim=-1, dtype=torch.float)
    from levelencoder import BinaryLevelEncoder
    encoder = BinaryLevelEncoder(784, 10000, 10, 0, 10)
    model = BinaryOnlineHD(10, encoder=encoder)
    print(model.diss(model.quantized_model()[0].unsqueeze(0)))
    mat = torch.randint(0, 2, (10, 10000), dtype=torch.int8)
    print(diss(mat[1], mat))
