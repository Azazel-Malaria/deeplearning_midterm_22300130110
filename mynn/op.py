from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None
        self.params = {'W' : self.W, 'b' : self.b}
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    def forward(self, X):
        self.input = X
        return np.dot(X, self.W) + self.b

    def backward(self, grad : np.ndarray):
        self.grads['W'] = np.dot(self.input.T, grad)
        self.grads['b'] = np.sum(grad, axis=0, keepdims=True)
        if self.weight_decay:
            self.grads['W'] += self.weight_decay_lambda * self.W
        grad_input = np.dot(grad, self.W.T)
        return grad_input
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

def xavier_init(shape):
    fan_in, fan_out = shape[1] * shape[2] * shape[3], shape[0]
    scale = np.sqrt(2.0 / (fan_in + fan_out))
    return np.random.normal(scale=scale, size=shape)

class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding = 0, initialize_method=np.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.W = initialize_method(size=(out_channels, in_channels, kernel_size, kernel_size)) * 0.1
        self.b = initialize_method(size=(out_channels, 1)) * 0.1
        self.grads = {'W': None, 'b': None}
        self.input = None
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda
        self.padding = padding
        self.params = {'W': self.W, 'b': self.b}

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    def _pad_input(self, X):
        if self.padding > 0:
            return np.pad(X, 
                        ((0,0), (0,0),
                        (self.padding, self.padding),
                        (self.padding, self.padding)),
                        mode='constant')
        return X

    def _im2col(self, X: np.ndarray, kernel_size: int, stride: int) -> np.ndarray:
        batch, channels, H, W = X.shape
        out_H = (H - kernel_size) // stride + 1
        out_W = (W - kernel_size) // stride + 1
        i0 = np.repeat(np.arange(kernel_size), kernel_size)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(out_H), out_W)
        j0 = np.tile(np.arange(kernel_size), kernel_size * channels)
        j1 = stride * np.tile(np.arange(out_W), out_H)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)
        k = np.repeat(np.arange(channels), kernel_size * kernel_size).reshape(-1, 1)
        cols = X[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(kernel_size * kernel_size * channels, -1)
        return cols.T
    def _col2im(self, cols, input_shape):
        batch_size, channels, H, W = input_shape
        out_H = (H - self.kernel_size) // self.stride + 1
        out_W = (W - self.kernel_size) // self.stride + 1
        cols_reshaped = cols.reshape(channels * self.kernel_size * self.kernel_size, -1, batch_size)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)
        img = np.zeros(input_shape)
        for b in range(batch_size):
            for c in range(channels):
                for i in range(self.kernel_size):
                    for j in range(self.kernel_size):
                        img[b, c, i:i + out_H * self.stride:self.stride, j:j + out_W * self.stride:self.stride] += \
                            cols_reshaped[b, c * self.kernel_size * self.kernel_size + i * self.kernel_size + j].reshape(out_H, out_W)
        return img
    
    def forward(self, X):
        X_pad = self._pad_input(X)
        self.input = X
        self.input_pad = X_pad
        batch_size, in_channels, in_H, in_W = X_pad.shape
        out_H = (in_H - self.kernel_size) // self.stride + 1
        out_W = (in_W - self.kernel_size) // self.stride + 1
        X_col = self._im2col(X_pad, self.kernel_size, self.stride)
        W_col = self.W.reshape(self.out_channels, -1).T
        output = np.dot(X_col, W_col) + self.b.T 
        output = output.reshape(batch_size, out_H, out_W, self.out_channels)
        output = output.transpose(0, 3, 1, 2)  
        return output
    def backward(self, d_out):
        X_pad = self.input_pad
        batch_size, _, out_H, out_W = d_out.shape
        d_out_reshaped = d_out.transpose(0, 2, 3, 1).reshape(-1, self.out_channels)
        W_col = self.W.reshape(self.out_channels, -1).T
        d_X_col = np.dot(d_out_reshaped, W_col.T)
        d_X_pad = self._col2im(d_X_col, X_pad.shape)
        X_col = self._im2col(X_pad, self.kernel_size, self.stride)
        d_W = np.dot(X_col.T, d_out_reshaped).T.reshape(self.W.shape)
        d_b = np.sum(d_out_reshaped, axis=0)
        if self.weight_decay:
            d_W += self.weight_decay_lambda * self.W
        self.grads['W'] = d_W
        self.grads['b'] = d_b.reshape(-1, 1)
        if self.padding > 0:
            d_X = d_X_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            d_X = d_X_pad
        del X_col, W_col, d_out_reshaped, d_X_col
        return d_X
    
    def clear_grad(self) -> None:
        self.grads = {'W': None, 'b': None}
        
class ReLU(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable =False
    def __call__(self, X):
        return self.forward(X)
    def forward(self, X):
        self.input = X
        return np.maximum(0, X)
    def backward(self, grads):
        assert self.input.shape == grads.shape
        return grads * (self.input > 0)
    
class MultiCrossEntropyLoss(Layer):
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.probs = None
        self.labels = None
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        if labels.ndim == 1:
            labels_onehot = np.eye(self.max_classes)[labels]
        else:
            labels_onehot = labels  
        if self.has_softmax:
            probs = softmax(predicts)
        else:
            probs = predicts
        self.probs = probs
        self.labels = labels_onehot
        epsilon = 1e-15 
        loss = -np.sum(labels_onehot * np.log(probs + epsilon)) / len(predicts)
        return loss

    def backward(self):
        if self.has_softmax:
            self.grads = (self.probs - self.labels) / len(self.probs)
        else:
            self.grads = -self.labels / (self.probs + 1e-15) / len(self.probs)
        self.model.backward(self.grads)
    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    def __init__(self, lambda_=1e-4):
        super().__init__()
        self.lambda_ = lambda_ 
        self.optimizable = False
        self.params = {}
    def forward(self, model):
        reg_loss = 0.0
        for layer in model.layers:
            if layer.optimizable and hasattr(layer, 'params'):
                for param in layer.params.values():
                    reg_loss += 0.5 * self.lambda_ * np.sum(param ** 2)
        return reg_loss
    def backward(self, model):
        for layer in model.layers:
            if layer.optimizable and hasattr(layer, 'params') and hasattr(layer, 'grads'):
                for param_name in layer.params.keys():
                    layer.grads[param_name] += self.lambda_ * layer.params[param_name]
       
def softmax(X):
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class CrossEntropyLossWithSoftmax(Layer):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        self.optimizable = False
        self.probs = None
        self.labels = None
    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        if labels.ndim == 1:
            labels_onehot = np.eye(predicts.shape[1])[labels]
        else:
            labels_onehot = labels
        max_vals = np.max(predicts, axis=1, keepdims=True)
        exp_scores = np.exp(predicts - max_vals)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.labels = labels_onehot
        epsilon = 1e-15 
        loss = -np.sum(labels_onehot * np.log(self.probs + epsilon)) / len(predicts)
        return loss
    
    def backward(self):
        batch_size = len(self.probs)
        grad = (self.probs - self.labels) / batch_size
        
        if self.model is not None:
            self.model.backward(grad)
        return grad
    
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False
    def __call__(self, model, current_loss):
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = []
                for layer in model.layers:
                    if hasattr(layer, 'params') and layer.optimizable:
                        self.best_weights.append(layer.params.copy())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    idx = 0
                    for layer in model.layers:
                        if hasattr(layer, 'params') and layer.optimizable:
                            layer.params.update(self.best_weights[idx])
                            idx += 1

    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        
class Dropout(Layer):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        self.mask = None
        self.optimizable = False

    def forward(self, X, training=True):
        if training:
            self.mask = (np.random.rand(*X.shape) > self.p) / (1 - self.p)
            return X * self.mask
        return X

    def backward(self, grad):
        return grad * self.mask 

    def __call__(self, X):
        return self.forward(X)
    
class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.optimizable = False
        self.input = None
        self.max_indices = None 

    def forward(self, X):
        self.input = X
        batch, channels, H_in, W_in = X.shape
        H_out = (H_in - self.kernel_size) // self.stride + 1
        W_out = (W_in - self.kernel_size) // self.stride + 1
        output = np.zeros((batch, channels, H_out, W_out))
        self.max_indices = np.zeros((batch, channels, H_out, W_out, 2), dtype=int)
        for b in range(batch):
            for c in range(channels):
                for h in range(H_out):
                    for w in range(W_out):
                        h_start = h * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        window = X[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.max(window)
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, h, w] = [h_start + max_idx[0], w_start + max_idx[1]]
        return output

    def backward(self, d_out):
        batch, channels, H_in, W_in = self.input.shape
        grad_input = np.zeros_like(self.input)
        for b in range(batch):
            for c in range(channels):
                for h in range(d_out.shape[2]):
                    for w in range(d_out.shape[3]):
                        max_h, max_w = self.max_indices[b, c, h, w]
                        grad_input[b, c, max_h, max_w] += d_out[b, c, h, w]
        return grad_input
    def __call__(self, X):
        return self.forward(X)