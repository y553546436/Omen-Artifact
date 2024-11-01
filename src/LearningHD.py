import torch

from learncoder import Encoder
from NN import LinearHDC
import torch.nn as nn

class LearningHD(nn.Module):
    def __init__(self, classes, features, dim=10000):
        super(LearningHD, self).__init__()
        self.encoder = Encoder(features, dim)
        self.model = LinearHDC(dim, classes)
        print(f"Encoder basis dtype: {self.encoder.basis.dtype}")
    
    def forward(self, x):
        return self.model(self.encoder(x))
    
    def encode(self, x):
        return self.encoder(x)
    
    def fit(self, x, y, x_test, y_test, epochs=100):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.05)
        criterion = nn.CrossEntropyLoss()
        
        epoch = 0
        last_loss = float('inf')
        while True:
            self.train()
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            self.eval()
            with torch.no_grad():
                output = self(x_test)
                test_loss = criterion(output, y_test)
                # get test accuracy
                y_pred = self.predict(x_test)
                acc = (y_pred == y_test).float().mean().item()
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()} - Test Loss: {test_loss.item()} - Test Accuracy: {acc}')
            
            epoch += 1
            if epochs > 0 and epoch >= epochs:
                break

            if -1e-6 < (last_loss - loss.item()) < 1e-6:
                break
            last_loss = loss.item()
        return self
    
    def predict(self, x):
        return self(x).argmax(1)
