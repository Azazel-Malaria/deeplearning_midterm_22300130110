from .op import *
import pickle
import mynn.op as nn

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout_rate = 0):
        super().__init__()
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)

class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.optimizable = False
        self.input_shape = None
    
    def forward(self, X):
        self.input_shape = X.shape
        return X.reshape(X.shape[0], -1) 
    
    def backward(self, d_out):
        return d_out.reshape(self.input_shape)
        
class Model_CNN(Layer):
    def __init__(self):
        super().__init__()
        self.layers = [
            nn.conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),
            nn.conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=2),
            Flatten(),
            nn.Linear(in_dim=64*7*7, out_dim=256),
            nn.ReLU(),
            nn.Linear(in_dim=256, out_dim=10)
        ]
        self.trainable_layers = []

    def __call__(self, X):
        return self.forward(X)
    
    def forward(self, X):
        if X.ndim == 3: 
            X = X[np.newaxis]
        elif X.ndim == 2: 
            X = X.reshape(-1, 1, 28, 28) 
            
        for layer in self.layers:
            X = layer.forward(X)
        return X
    
    def backward(self, d_out):
        for layer in reversed(self.layers):
            d_out = layer.backward(d_out)
        return d_out

    def load_model(self, param_list) -> None:
        if isinstance(param_list, str):
            param_list = np.load(param_list, allow_pickle=True)
        
        param_idx = 0
        for layer in self.trainable_layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.W = param_list[param_idx]
                layer.b = param_list[param_idx + 1]
                param_idx += 2
    def save_model(self, save_path: str) -> None:
        param_list = []
        for layer in self.trainable_layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                param_list.append(layer.W)
                param_list.append(layer.b)
        np.save(save_path, param_list)
