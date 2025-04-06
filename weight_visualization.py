# codes to make visualization of your weights.
import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

from pathlib import Path

def load_model_and_data(model_path):
    try:
        model = nn.models.Model_MLP()
        model.load_model(model_path)
        model_type = "MLP"
    except:
        model = nn.models.Model_CNN()
        model.load_model(model_path)
        model_type = "CNN"
    test_images_path = './dataset/MNIST/t10k-images-idx3-ubyte.gz'
    test_labels_path = './dataset/MNIST/t10k-labels-idx1-ubyte.gz'
    with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        test_imgs = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)
    test_imgs = test_imgs / test_imgs.max()
    return model, test_imgs, test_labs, model_type

def plot_original_weights(model, model_type):
    if model_type == "MLP":
        weight_layers = [layer for layer in model.layers if hasattr(layer, 'params')]
        
        if len(weight_layers) >= 1:
            W1 = weight_layers[0].params['W']
            n_units = min(30, W1.shape[1])
            plt.figure(figsize=(15, 10))
            plt.suptitle('First Layer Weights Visualization', y=1.02)
            n_cols = 6
            n_rows = int(np.ceil(n_units / n_cols))
            for i in range(n_units):
                plt.subplot(n_rows, n_cols, i+1)
                plt.imshow(W1.T[i].reshape(28, 28), 
                          cmap='viridis', 
                          interpolation='nearest')
                plt.title(f'Unit {i}', fontsize=8)
                plt.xticks([])
                plt.yticks([])
            plt.tight_layout()
            plt.show()
        if len(weight_layers) >= 2:
            W2 = weight_layers[1].params['W']
            plt.figure(figsize=(12, 8))
            plt.imshow(W2, cmap='coolwarm', aspect='auto')
            plt.colorbar(label='Weight Value')
            plt.title('Second Layer Weight Matrix')
            plt.xlabel('Output Units')
            plt.ylabel('Input Units')
            plt.show()
    else:
        print("Note: Original weight visualization is designed for MLP models")

def plot_conv_weights(model):
    conv_layers = [layer for layer in model.layers if isinstance(layer, nn.op.conv2D)]
    for layer_idx, layer in enumerate(conv_layers):
        weights = layer.params['W']
        out_channels, in_channels = weights.shape[:2]
        print(f"\nVisualizing Conv Layer {layer_idx+1}")
        print(f"Shape: {weights.shape} (out_channels, in_channels, height, width)")
        plt.figure(figsize=(15, 2))
        plt.suptitle(f'Conv Layer {layer_idx+1} - First Output Channel Across All Input Channels', y=1.05)
        for ic in range(min(in_channels, 16)): 
            plt.subplot(1, min(in_channels, 16), ic+1)
            kernel = weights[0, ic] 
            plt.imshow(kernel, cmap='viridis')
            plt.title(f'In {ic}', fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(15, 2))
        plt.suptitle(f'Conv Layer {layer_idx+1} - First Input Channel Across All Output Channels', y=1.05)
        
        for oc in range(min(out_channels, 16)):
            plt.subplot(1, min(out_channels, 16), oc+1)
            kernel = weights[oc, 0]
            plt.imshow(kernel, cmap='viridis')
            plt.title(f'Out {oc}', fontsize=8)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

def plot_accuracy_curves(history_path='./best_models/training_history.pickle'):
    with open(history_path, 'rb') as f:
        history = pickle.load(f)
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy', alpha=0.7)
    plt.plot(history['val_acc'], label='Validation Accuracy', alpha=0.7)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    model_path = './saved_models/addhidden_MLP.pickle'
    model, test_imgs, test_labs, model_type = load_model_and_data(model_path)
    print(f"Loaded {model_type} model")
    plot_original_weights(model, model_type)
    if model_type == "CNN":
        plot_conv_weights(model)
    plot_accuracy_curves()