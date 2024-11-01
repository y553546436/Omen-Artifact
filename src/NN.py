import torch
import torch.nn as nn
import torch.nn.functional as F
from LDC import ClassLayer


class LinearHDC(nn.Module):
    def __init__(self, in_features, out_features, encoder=None):
        super(LinearHDC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = encoder
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, x):
        return torch.log_softmax(F.linear(x, self.weight), dim=-1)
    
    def fit(self, x, y, x_test, y_test, epochs=-1):
        optimizer = torch.optim.Adam(self.parameters())
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


class BinaryLinearHD(nn.Module):
    def __init__(self, in_features, out_features, encoder=None):
        super(BinaryLinearHD, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = encoder
        self.binary_layer = ClassLayer(in_features, out_features)
    
    def forward(self, x):
        return self.binary_layer(x)
    
    def encode(self, x):
        return self.encoder(x)
    
    def predict(self, x):
        return self(x).argmax(1)
    
    def fit(self, x, y, x_test, y_test, epochs=500):
        optimizer = torch.optim.Adam(self.parameters())
        criterion = torch.nn.CrossEntropyLoss()
        
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
                y_pred = output.argmax(1)
                acc = (y_pred == y_test).float().mean().item()
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()} - Test Loss: {test_loss.item()} - Test Accuracy: {acc}')
            
            epoch += 1
            if epoch >= epochs:
                break

            if -1e-6 < (last_loss - loss.item()) < 1e-6:
                break
            last_loss = loss.item()
        return self
    