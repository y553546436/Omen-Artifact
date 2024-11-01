import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from string import Template
from utils import tensor_to_c_array
import pickle


class BinaryActivation(torch.nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        out_forward = torch.sign(out_forward + 0.5)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3
        return out

class ValueBox(nn.Module):
    def __init__(self, D):
        super(ValueBox, self).__init__()
        self.dimension = D
        self.fc1 = nn.Linear(1, 20, bias=True)
        self.bn = nn.BatchNorm1d(20)
        self.act = nn.Tanh()
        self.fc3 = nn.Linear(20, self.dimension, bias = True)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(self.bn(x))
        x = self.fc3(x)
        return x

class FeatureLayer(nn.Module):
    def __init__(self, N, FD, VD):
        super(FeatureLayer, self).__init__()
        self.num_feature = N
        self.fhv_dimension = FD
        self.vhv_dimension = VD
        self.rept = int(FD / VD)
        self.weight = torch.nn.Parameter(torch.empty((self.num_feature, self.fhv_dimension)))
        nn.init.xavier_uniform_(self.weight)
        self.scaling_factor = 0
        
    def forward(self, x):
        x = x.view(-1,self.num_feature,self.vhv_dimension).repeat(1,1,self.rept)
        real_weights = self.weight
        self.scaling_factor = torch.mean(abs(real_weights))
        self.scaling_factor = self.scaling_factor.detach()
        binary_weights_no_grad = self.scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = torch.sum(x * binary_weights, dim=1)
        return y

class ClassLayer(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(ClassLayer, self).__init__()
        self.shape = (out_shape, in_shape)
        self.weight = torch.nn.Parameter(torch.empty(self.shape))
        nn.init.xavier_uniform_(self.weight)
        self.scaling_factor = 0

    def forward(self, x):
        real_weights = self.weight
        self.scaling_factor = torch.mean(abs(real_weights))
        self.scaling_factor = self.scaling_factor.detach()
        binary_weights_no_grad = self.scaling_factor * torch.sign(real_weights)
        cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.linear(x, binary_weights)
        return y

class LDC(nn.Module):
    def __init__(self, features, classes, feature_hv_dim, value_hv_dim):
        super(LDC, self).__init__()
        self.num_feature = features
        self.fhv_dimension = feature_hv_dim
        self.vhv_dimension = value_hv_dim
        self.num_class = classes
        self.value_box = ValueBox(self.vhv_dimension)
        self.feature_layer = FeatureLayer(self.num_feature, self.fhv_dimension, self.vhv_dimension)
        self.binarization = BinaryActivation()
        self.class_layer = ClassLayer(self.fhv_dimension, self.num_class)

    def forward(self, x):
        x = self.encode(x)
        x = self.class_layer(x)
        return x
    
    def encode_batch(self, x):
        x = x.reshape(-1, 1)
        x = self.value_box(x)
        Val_Out = x.view(-1,self.num_feature*self.vhv_dimension)
        x = self.binarization(Val_Out)
        L_in = self.feature_layer(x)
        x = self.binarization(L_in)
        return x
    
    def encode(self, x):
        # do encode in batch to avoid memory error
        batch_size = 1000
        encoded = torch.empty((x.size(0), self.fhv_dimension), device=x.device)
        for i in range(0, x.size(0), batch_size):
            encoded[i:i+batch_size] = self.encode_batch(x[i:i+batch_size])
        return encoded
    
    def fit(self, x, y, x_test, y_test, epochs=100, batch_size=1000, lr=1e-4):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        epoch = 0
        last_loss = float('inf')
        while True:
            self.train()
            total_loss = 0

            for i in tqdm(range(0, x.size(0), batch_size)):
                x_batch = x[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                output = self(x_batch)
                loss = criterion(output, y_batch)
                total_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                # clip gradient
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
            
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                total_test_loss = 0
                for i in tqdm(range(0, x_test.size(0), batch_size)):
                    x_test_batch = x_test[i:i+batch_size]
                    y_test_batch = y_test[i:i+batch_size]
                    output = self(x_test_batch)
                    test_loss = criterion(output, y_test_batch)
                    total_test_loss += test_loss.item()
                    y_pred_batch = output.argmax(1)
                    correct += (y_pred_batch == y_test_batch).sum().item()
                    total += y_test_batch.size(0)
                acc = correct / total
            print(f'Epoch {epoch + 1}/{epochs} - Loss: {total_loss} - Test Loss: {total_test_loss} - Test Accuracy: {acc}')
            
            epoch += 1
            if epochs > 0 and epoch >= epochs:
                break

            if 1e-6 < (last_loss - total_loss) < 1e-6:
                break
            last_loss = total_loss
        return self
    
    def predict(self, x):
        return self(x).argmax(1)
    

    def generate_header(self):
        header = """ // This file is generated from the LDC model
#ifndef LDC_MODEL_H
#define LDC_MODEL_H

#define BINARY

#include <stdint.h>

#define NUMFEATURE $num_feature
#define NUMCLASS $num_class
#define FEATURE_HV_DIM $fhv_dimension
#define VALUE_HV_DIM $vhv_dimension

#define VALUE_BOX_HIDDEN_DIM $vb_hidden_dim

// Value Box Parameters
const float vb_bn_running_mean[VALUE_BOX_HIDDEN_DIM] = $vb_bn_running_mean;
const float vb_bn_gamma[VALUE_BOX_HIDDEN_DIM] = $vb_bn_gamma;
const float vb_bn_beta[VALUE_BOX_HIDDEN_DIM] = $vb_bn_beta;

const float value_box_fc1_weight[VALUE_BOX_HIDDEN_DIM] = $vb_fc1_weight;
const float value_box_fc1_bias[VALUE_BOX_HIDDEN_DIM] = $vb_fc1_bias;
const float value_box_fc3_weight[VALUE_HV_DIM][VALUE_BOX_HIDDEN_DIM] = $vb_fc3_weight;
const float value_box_fc3_bias[VALUE_HV_DIM] = $vb_fc3_bias;

// Feature Layer Parameters
const int feature_layer_weight[FEATURE_HV_DIM][NUMFEATURE] = $feature_layer_weight;

const uint32_t class_layer_weight[FEATURE_HV_DIM][NUMCLASS] = $class_layer_weight;

#endif
"""
        header_template = Template(header)
        data_mapping = {
            'num_feature': self.num_feature,
            'num_class': self.num_class,
            'fhv_dimension': self.fhv_dimension,
            'vhv_dimension': self.vhv_dimension,
            'vb_hidden_dim': self.value_box.fc1.out_features,
            'vb_bn_running_mean': self.value_box.bn.running_mean,
            'vb_bn_gamma': self.value_box.bn.weight / torch.sqrt(self.value_box.bn.running_var + self.value_box.bn.eps),
            'vb_bn_beta': self.value_box.bn.bias,
            'vb_fc1_weight': self.value_box.fc1.weight.flatten(),
            'vb_fc1_bias': self.value_box.fc1.bias,
            'vb_fc3_weight': self.value_box.fc3.weight,
            'vb_fc3_bias': self.value_box.fc3.bias,
            'feature_layer_weight': bipolar_to_binary(self.feature_layer.weight.T.sign().int()),
            'class_layer_weight': bipolar_to_binary(self.class_layer.weight.T.sign().int()),
        }
        shapes = {
            'num_feature': data_mapping['num_feature'],
            'num_class': data_mapping['num_class'],
            'fhv_dimension': data_mapping['fhv_dimension'],
            'vhv_dimension': data_mapping['vhv_dimension'],
            'vb_hidden_dim': data_mapping['vb_hidden_dim'],
            'vb_bn_running_mean': data_mapping['vb_bn_running_mean'].shape,
            'vb_bn_gamma': data_mapping['vb_bn_gamma'].shape,
            'vb_bn_beta': data_mapping['vb_bn_beta'].shape,
            'vb_fc1_weight': data_mapping['vb_fc1_weight'].shape,
            'vb_fc1_bias': data_mapping['vb_fc1_bias'].shape,
            'vb_fc3_weight': data_mapping['vb_fc3_weight'].shape,
            'vb_fc3_bias': data_mapping['vb_fc3_bias'].shape,
            'feature_layer_weight': data_mapping['feature_layer_weight'].shape,
            'class_layer_weight': data_mapping['class_layer_weight'].shape,
        }
        print(shapes)
        string_mapping = {
            'num_feature': f'{shapes["num_feature"]}',
            'num_class': f'{shapes["num_class"]}',
            'fhv_dimension': f'{shapes["fhv_dimension"]}',
            'vhv_dimension': f'{shapes["vhv_dimension"]}',
            'vb_hidden_dim': f'{shapes["vb_hidden_dim"]}',
            'vb_bn_running_mean': tensor_to_c_array(data_mapping['vb_bn_running_mean']),
            'vb_bn_gamma': tensor_to_c_array(data_mapping['vb_bn_gamma']),
            'vb_bn_beta': tensor_to_c_array(data_mapping['vb_bn_beta']),
            'vb_fc1_weight': tensor_to_c_array(data_mapping['vb_fc1_weight']),
            'vb_fc1_bias': tensor_to_c_array(data_mapping['vb_fc1_bias']),
            'vb_fc3_weight': tensor_to_c_array(data_mapping['vb_fc3_weight']),
            'vb_fc3_bias': tensor_to_c_array(data_mapping['vb_fc3_bias']),
            'feature_layer_weight': tensor_to_c_array(data_mapping['feature_layer_weight']),
            'class_layer_weight': tensor_to_c_array(data_mapping['class_layer_weight']),
        }
        header_content = header_template.substitute(string_mapping)
        return header_content


def bipolar_to_binary(x):
    return ((-torch.sign(x) + 1) / 2).long()


def generate_header(args, data_dir):
    model_args = pickle.load(open(f'{data_dir}/ldc_model_args.pkl', 'rb'))
    model = LDC(**model_args)
    model.load_state_dict(torch.load(f'{data_dir}/ldc_model_state.pth', map_location=torch.device('cpu')))
    header = model.generate_header()
    return header


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='LDC Model')
    parser.add_argument('--dir', type=str, help='Directory to read the trained model', required=True)
    parser.add_argument('--header', type=str, help='Path to save the generated header file', required=True)
    args = parser.parse_args()
    with open(f'{args.header}', 'w') as f:
        f.write(generate_header(args, args.dir))
