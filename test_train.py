# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot
from scipy.ndimage import rotate, shift, zoom
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
np.random.seed(309)

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.
idx = np.random.permutation(np.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()

train_imgs = train_imgs.astype(np.float32)
valid_imgs = valid_imgs.astype(np.float32)

###################augment dataset#################
def augment_image(image, max_shift=2, max_rotate=15, max_zoom=0.1):
    dx, dy = np.random.randint(-max_shift, max_shift+1, 2)
    angle = np.random.uniform(-max_rotate, max_rotate)
    scale = 1 + np.random.uniform(-max_zoom, max_zoom)
    transformed = image.reshape(28, 28)
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
    return transformed.reshape(-1)
def augmented_train_data(X, y, augment_factor=3):
    augmented_X = []
    augmented_y = []
    for img, label in zip(X, y):
        augmented_X.append(img)
        augmented_y.append(label)
        for _ in range(augment_factor):
            augmented_X.append(augment_image(img))
            augmented_y.append(label)
    return np.array(augmented_X), np.array(augmented_y)
train_imgs_aug, train_labs_aug = augmented_train_data(train_imgs, train_labs)
###########################augment dataset####################################


linear_model = nn.models.Model_MLP([train_imgs.shape[-1], 512, 256, 128, 10], 'ReLU')
CNN = nn.models.Model_CNN()
model = linear_model
optimizer = nn.optimizer.MomentGD(init_lr=0.06, model=model)
scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
loss_fn = nn.op.CrossEntropyLossWithSoftmax(model=model)
early_stopping = nn.op.EarlyStopping(
    patience=5,  
    min_delta=0.001, 
    restore_best_weights=True 
)
runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler, early_stopping=early_stopping)
print("Initialization finished")
def data_generator(X, y, batch_size=64, augment_factor=3):
    n_samples = len(X)
    while True:
        indices = np.random.permutation(n_samples)
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i + batch_size]
            X_batch = []
            y_batch = []
            for idx in batch_indices:
                X_batch.append(X[idx])
                y_batch.append(y[idx])
                for _ in range(augment_factor):
                    X_batch.append(augment_image(X[idx]))
                    y_batch.append(y[idx])
            yield np.array(X_batch), np.array(y_batch)
train_gen = data_generator(train_imgs, train_labs, batch_size=64)
runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=100, save_dir=r'./best_models')
_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.show()
final_model_path = './saved_models/transition_MLP.pickle'
runner.model.save_model(final_model_path)
print(f"model saved to {final_model_path}")
history = {
    'train_acc': runner.train_scores,
    'val_acc': runner.dev_scores,
    'train_loss': runner.train_loss,
    'val_loss': runner.dev_loss
}
with open('./saved_models/transition_MLP_history.pickle', 'wb') as f:
    pickle.dump(history, f)
print("trained model has been saved")