import mindspore as ms
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from mindspore import Tensor
import mindspore.nn as nn

def load_mindspore_model(model_path):
    with open(model_path, 'rb') as f:
        weights_dict = pickle.load(f)
    return weights_dict

def plot_conv_weights(weights_dict, layer_name, n_filters=16):
    if layer_name not in weights_dict:
        print(f"Warning: {layer_name} not found in weights dictionary")
        return
    
    weights = weights_dict[layer_name]
    out_channels, in_channels, h, w = weights.shape
    n_filters = min(n_filters, out_channels)
    plt.figure(figsize=(15, 2))
    plt.suptitle(f'{layer_name} - First {n_filters} Filters', y=1.05)
    for i in range(n_filters):
        for j in range(in_channels):
            plt.subplot(n_filters, in_channels, i*in_channels + j + 1)
            kernel = weights[i, j]
            plt.imshow(kernel, cmap='viridis')
            plt.title(f'F{i}I{j}', fontsize=6)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_feature_maps(model_weights, input_image, layer_names):
    class FeatureExtractor(nn.Cell):
        def __init__(self, weights_dict):
            super(FeatureExtractor, self).__init__()
            self.conv1 = nn.Conv2d(1, 6, kernel_size=5, pad_mode='valid')
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, pad_mode='valid')
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.conv1.weight = ms.Parameter(Tensor(model_weights['network.conv1.weight']))
            self.conv2.weight = ms.Parameter(Tensor(model_weights['network.conv2.weight']))
        
        def construct(self, x):
            outputs = {}
            x = self.conv1(x)
            outputs['conv1'] = x
            x = self.relu1(x)
            outputs['relu1'] = x
            x = self.pool1(x)
            outputs['pool1'] = x
            x = self.conv2(x)
            outputs['conv2'] = x
            x = self.relu2(x)
            outputs['relu2'] = x
            x = self.pool2(x)
            outputs['pool2'] = x
            return outputs
    extractor = FeatureExtractor(model_weights)
    input_tensor = Tensor(input_image, ms.float32)
    feature_maps = extractor(input_tensor)
    for layer_name in layer_names:
        if layer_name in feature_maps:
            maps = feature_maps[layer_name].asnumpy()
            num_filters = maps.shape[1]
            plt.figure(figsize=(15, 2))
            plt.suptitle(f'Feature Maps: {layer_name}', y=1.05)
            for i in range(min(16, num_filters)):
                plt.subplot(2, 8, i+1)
                plt.imshow(maps[0, i], cmap='viridis')
                plt.title(f'F{i}', fontsize=8)
                plt.axis('off')
            plt.tight_layout()
            plt.show()

def plot_training_history(history_path):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()

def visualize_random_sample(dataset_path):
    train_images_path = os.path.join(dataset_path, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(dataset_path, 'train-labels-idx1-ubyte.gz')
    mnist_dataset = MNISTDataset(train_images_path, train_labels_path)
    idx = np.random.randint(len(mnist_dataset))
    image, label = mnist_dataset[idx]
    plt.figure()
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    plt.show()
    return image.reshape(1, 1, 28, 28)

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

if __name__ == "__main__":
    import gzip
    from struct import unpack
    ms.set_seed(42)
    model_path = './saved_models/LeNet_trans.pickle'
    history_path = './saved_models/LeNet_trans_history.pickle'
    dataset_path = './dataset/MNIST/'
    weights_dict = load_mindspore_model(model_path)
    print("All weight dict keys:", weights_dict.keys())
    if os.path.exists(history_path):
        plot_training_history(history_path)
    plot_conv_weights(weights_dict, 'network.conv1.weight', n_filters=6)
    plot_conv_weights(weights_dict, 'network.conv2.weight', n_filters=16)
    sample_image = visualize_random_sample(dataset_path)
    plot_feature_maps(weights_dict, sample_image, ['conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2'])