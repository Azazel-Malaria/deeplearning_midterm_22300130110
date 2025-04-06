from abc import abstractmethod
import numpy as np


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass

class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, momentum=0.9):
        super().__init__(init_lr, model)
        self.momentum = momentum 
        self.velocities = {}
        for layer in self.model.layers:
            if layer.optimizable:
                for param_name in layer.params.keys():
                    self.velocities[(id(layer), param_name)] = np.zeros_like(layer.params[param_name])
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable:
                for param_name in layer.params.keys():
                    param = layer.params[param_name]
                    grad = layer.grads[param_name]
                    velocity_key = (id(layer), param_name)
                    self.velocities[velocity_key] = self.momentum * self.velocities[velocity_key] - self.init_lr * grad
                    if layer.weight_decay:
                        param *= (1 - self.init_lr * layer.weight_decay_lambda)
                    param += self.velocities[velocity_key]
                    layer.params[param_name] = param