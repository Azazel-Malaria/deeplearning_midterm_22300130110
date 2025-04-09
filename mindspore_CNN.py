import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
import mindspore.dataset.vision as vision
import mindspore.dataset.transforms as transforms
from mindspore import Tensor, Model
from mindspore.common.initializer import Normal
from mindspore.train.callback import Callback, LossMonitor, TimeMonitor
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift, zoom
import gzip
from struct import unpack
import pickle
import os
ms.set_seed(309)

class MNISTDataset:
    def __init__(self, images_path, labels_path):
        with gzip.open(images_path, 'rb') as f:
            magic, num, rows, cols = unpack('>4I', f.read(16))
            self.images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28, 28, 1)
        with gzip.open(labels_path, 'rb') as f:
            magic, num = unpack('>2I', f.read(8))
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)
        self.images = self.images.astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

class DataAugmentation:
    def __init__(self, max_shift=2, max_rotate=15, max_zoom=0.1):
        self.max_shift = max_shift
        self.max_rotate = max_rotate
        self.max_zoom = max_zoom
    
    def __call__(self, image):
        dx, dy = np.random.randint(-self.max_shift, self.max_shift+1, 2)
        angle = np.random.uniform(-self.max_rotate, self.max_rotate)
        scale = 1 + np.random.uniform(-self.max_zoom, self.max_zoom)
        transformed = image.squeeze()
        transformed = shift(transformed, [dy, dx], mode='nearest')
        transformed = rotate(transformed, angle, reshape=False, mode='nearest')
        transformed = zoom(transformed, scale, mode='nearest')
        if scale != 1:
            h, w = transformed.shape
            if h > 28:
                start = (h - 28) // 2
                transformed = transformed[start:start+28, start:start+28]
            else:
                padded = np.zeros((28, 28))
                start = (28 - h) // 2
                padded[start:start+h, start:start+h] = transformed
                transformed = padded
        return transformed.reshape(28, 28, 1).astype(np.float32)

class CNN(nn.Cell):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, pad_mode='pad', weight_init='normal')
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, pad_mode='pad', weight_init='normal')
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(64*7*7, 256, weight_init='normal')
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Dense(256, 10, weight_init='normal')
    
    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x

class LeNet(nn.Cell):
    def __init__(self):
        super(LeNet, self).__init__()
        # Original LeNet uses 32x32 input, but commonly adapted for 28x28 (MNIST)
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid', weight_init=Normal(0.02))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid', weight_init=Normal(0.02))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flatten = nn.Flatten()
        # For MNIST 28x28 input: ((28-4)/2-4)/2 = 4x4 output before flatten
        self.fc1 = nn.Dense(16*4*4, 120, weight_init=Normal(0.02))
        self.relu3 = nn.ReLU()
        
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.relu4 = nn.ReLU()
        
        self.fc3 = nn.Dense(84, 10, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        
        x = self.fc2(x)
        x = self.relu4(x)
        
        x = self.fc3(x)
        return x

class CustomTrainStep(nn.TrainOneStepCell):
    def __init__(self, network, optimizer):
        super(CustomTrainStep, self).__init__(network, optimizer)
        self.loss_scale = 1.0

    def construct(self, data, label):
        loss = self.network(data, label) 
        sens = ops.fill(loss.dtype, loss.shape, self.loss_scale) 
        grads = self.grad(self.network, self.weights)(data, label, sens)
        grads = self.grad_reducer(grads)
        loss = ops.depend(loss, self.optimizer(grads))
        return loss

class EarlyStopping(Callback):
    def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.early_stop = False
    
    def on_train_begin(self, run_context):
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        current_loss = cb_params.net_outputs.asnumpy()
        
        if (self.best_loss - current_loss) > self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = [param.copy() for param in cb_params.train_network.get_parameters()]
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    for param, best_param in zip(cb_params.train_network.get_parameters(), self.best_weights):
                        param.set_data(best_param)
                run_context.request_stop()

def augment_image(image_np):
    augmented = DataAugmentation()(image_np)
    return augmented.transpose(2, 0, 1).astype(np.float32) 

def create_dataset(images, labels, batch_size=64, augment=False, shuffle=True):
    dataset = ds.GeneratorDataset(
        source=list(zip(images, labels)), 
        column_names=["image", "label"],
        shuffle=shuffle
    )
    transform_img = [
        vision.Rescale(1.0 / 255.0, 0.0),
        vision.HWC2CHW(),
        transforms.TypeCast(ms.float32)
    ]
    transform_label = [transforms.TypeCast(ms.int32)]
    dataset = dataset.map(operations=transform_img, input_columns="image")
    dataset = dataset.map(operations=transform_label, input_columns="label")
    if augment:
        dataset = dataset.map(
            operations=lambda x: augment_image(x),
            input_columns="image",
            python_multiprocessing=True
        )
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

class HistoryRecorder(Callback):
    def __init__(self, eval_network, valid_dataset):
        super(HistoryRecorder, self).__init__()
        self.eval_network = eval_network
        self.valid_dataset = valid_dataset
        self.train_acc = []
        self.val_acc = []
        self.train_loss = []
        self.val_loss = []

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        # 记录训练损失
        self.train_loss.append(float(cb_params.net_outputs.asnumpy()))
        
        # 计算训练准确率（可选）
        train_metrics = self._calculate_metrics(cb_params.train_dataset, self.eval_network)
        self.train_acc.append(train_metrics['accuracy'])
        
        # 计算验证集指标
        val_metrics = self._calculate_metrics(self.valid_dataset, self.eval_network)
        self.val_loss.append(val_metrics['loss'])
        self.val_acc.append(val_metrics['accuracy'])

    def _calculate_metrics(self, dataset, eval_network):
        metrics = {
            'accuracy': nn.Accuracy(),
            'loss': nn.Loss()
        }
        for batch in dataset.create_dict_iterator():
            data = batch['image']
            label = batch['label']
            pred, _, _ = eval_network(data, label)
            loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')(pred, label)
            metrics['accuracy'].update(pred, label)
            metrics['loss'].update(loss)
        return {
            'accuracy': metrics['accuracy'].eval(),
            'loss': metrics['loss'].eval()
        }

    def get_history(self):  # 确保此方法存在且缩进正确
        return {
            'train_acc': self.train_acc,
            'val_acc': self.val_acc,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss
        }
def save_model_and_history(model, history, model_path='./saved_models/LeNet.pickle', 
                          history_path='./saved_models/LeNet_history.pickle'):
    # 确保保存目录存在
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # 保存模型权重为pickle格式
    weights_dict = {param.name: param.data.asnumpy() for param in model.train_network.network.get_parameters()}
    with open(model_path, 'wb') as f:
        pickle.dump(weights_dict, f)
    
    # 保存训练历史
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"Model weights saved to {model_path}")
    print(f"Training history saved to {history_path}")

def train():
    # 数据加载和预处理
    train_images_path = './dataset/MNIST/train-images-idx3-ubyte.gz'
    train_labels_path = './dataset/MNIST/train-labels-idx1-ubyte.gz'
    mnist_dataset = MNISTDataset(train_images_path, train_labels_path)
    num_samples = len(mnist_dataset)
    indices = np.random.permutation(np.arange(num_samples))
    train_imgs = mnist_dataset.images[indices[10000:]]
    train_labs = mnist_dataset.labels[indices[10000:]]
    valid_imgs = mnist_dataset.images[indices[:10000]]
    valid_labs = mnist_dataset.labels[indices[:10000]]
    
    train_dataset = create_dataset(train_imgs, train_labs, batch_size=64, augment=True)
    valid_dataset = create_dataset(valid_imgs, valid_labs, batch_size=64, augment=False)

    # 模型初始化
    model = LeNet()

    # 损失函数和优化器
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    optimizer = nn.Momentum(model.trainable_params(), learning_rate=0.06, momentum=0.9)

    # 定义评估网络
    class EvalNet(nn.Cell):
        def __init__(self, network):
            super(EvalNet, self).__init__()
            self.network = network
        
        def construct(self, data, label):
            pred = self.network(data)
            return pred, label, 0  # 保持三输出格式
    # 创建模型
    net_with_loss = nn.WithLossCell(model, loss_fn)
    train_net = CustomTrainStep(net_with_loss, optimizer)
    eval_net = EvalNet(model)
    model = Model(
        network=train_net,
        eval_network=eval_net,
        metrics={'accuracy': nn.Accuracy()},
        eval_indexes=[0, 1, 2]  # 必须保持长度为3
    )

    # 初始化回调
    # 初始化回调时传入验证数据集
    history_recorder = HistoryRecorder(eval_net, valid_dataset)
    callbacks = [
        LossMonitor(), 
        TimeMonitor(), 
        history_recorder
    ]
    
    # 训练模型
    model.train(5, train_dataset, callbacks=callbacks, dataset_sink_mode=False)
    
    # 获取历史记录
    history = history_recorder.get_history()  # 确保调用正确
    
    # 保存和绘图
    save_model_and_history(model, history)
    plt.show()
    # 保存模型和历史
    save_model_and_history(model, history)

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train()