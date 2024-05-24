import numpy as np
import scipy.io as sio
import scipy.misc
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import confusion_matrix

input_dimension = 102
num_classes     = 9
window_size     = 27
latent_dim      = 64 # Latent dimension for the autoencoder

# Autoencoder model
def create_autoencoder(input_dim, latent_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation='relu')(input_layer)
    encoded = layers.Dense(latent_dim, activation='relu')(encoded)
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)
    autoencoder = models.Model(input_layer, decoded)
    encoder = models.Model(input_layer, encoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

# Load and normalize data
data_mat = sio.loadmat('D:/SSAN/Pavia.mat')
data_in  = data_mat['pavia']
mat_gt   = sio.loadmat('D:/SSAN/Pavia_gt.mat')
label    = mat_gt['pavia_gt']
GT       = label.reshape(np.prod(label.shape[:2]),)

labeled_sets = np.load('D:/SSAN/labeled_index.npy')
valid_sets   = np.load('D:/SSAN/valid_index.npy')
test_sets    = np.load('D:/SSAN/test_index.npy')
all_sets     = np.load('D:/SSAN/all_index.npy')

normdata = np.zeros((data_in.shape[0], data_in.shape[1], data_in.shape[2]), dtype=np.float32)

for dim in range(data_in.shape[2]):
    normdata[:, :, dim] = (data_in[:, :, dim] - np.amin(data_in[:, :, dim])) / \
                          float((np.amax(data_in[:, :, dim]) - np.amin(data_in[:, :, dim])))

# Prepare data for autoencoder
data_flat = normdata.reshape(-1, normdata.shape[-1])

# Train autoencoder
autoencoder, encoder = create_autoencoder(input_dim=normdata.shape[-1], latent_dim=latent_dim)
autoencoder.fit(data_flat, data_flat, epochs=50, batch_size=256, shuffle=True, validation_split=0.1)

# Transform data using the trained encoder
data_encoded = encoder.predict(data_flat)
data_encoded = data_encoded.reshape(normdata.shape[0], normdata.shape[1], latent_dim)

# Normalize encoded data
normpca = np.zeros((data_encoded.shape[0], data_encoded.shape[1], data_encoded.shape[2]), dtype=np.float32)
for dim in range(data_encoded.shape[2]):
    normpca[:, :, dim] = (data_encoded[:, :, dim] - np.amin(data_encoded[:, :, dim])) / \
                          float((np.amax(data_encoded[:, :, dim]) - np.amin(data_encoded[:, :, dim])))

# Pad data
margin = int((window_size - 1) / 2)
def PadWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

padded_data = PadWithZeros(normpca, margin=margin)

# One-hot encoding
def DenseToOneHot(labels_dense, num_classes=16):
    num_labels     = labels_dense.shape[0]
    index_offset   = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel() - 1] = 1
    return labels_one_hot

# DataSet class
class DataSet(object):
    def __init__(self, images):
        self._num_examples = images.shape[0]
        self._images = images
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._epochs_completed += 1
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        
        hsi_batch_pca = np.zeros((batch_size, window_size, window_size, latent_dim), dtype=np.float32)
        hsi_batch_patch = np.zeros((batch_size, input_dimension), dtype=np.float32)
        col_pca = data_encoded.shape[1]
        col     = data_in.shape[1]
        for q1 in range(batch_size):
            hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col), (self._images[start + q1] % col), :]
            hsi_batch_pca[q1]   = padded_data[(self._images[start + q1] // col_pca):
                                              ((self._images[start + q1] // col_pca) + window_size),
                                              (self._images[start + q1] % col_pca):
                                              ((self._images[start + q1] % col_pca) + window_size), :]
        block = self._images[start:end]
        hsi_batch_label = GT[block]
        hsi_batch_label = DenseToOneHot(hsi_batch_label, num_classes=num_classes)
        return hsi_batch_patch, hsi_batch_pca, hsi_batch_label

    def next_batch_test(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            self._index_in_epoch = self._num_examples
        end = self._index_in_epoch
        hsi_batch_pca = np.zeros((end-start, window_size, window_size, latent_dim), dtype=np.float32)
        col_pca = data_encoded.shape[1]
        hsi_batch_patch = np.zeros((end-start, input_dimension), dtype=np.float32)
        col = data_in.shape[1]
        for q1 in range(end-start):
            hsi_batch_patch[q1] = normdata[(self._images[start + q1] // col),(self._images[start + q1] % col),:]
            hsi_batch_pca[q1] = padded_data[(self._images[start + q1] // col_pca):
                                            ((self._images[start + q1] // col_pca) + window_size),
                                            (self._images[start + q1] % col_pca):
                                            ((self._images[start + q1] % col_pca) + window_size), :]
        block = self._images[start:end]
        hsi_batch_label = GT[block]
        hsi_batch_label = DenseToOneHot(hsi_batch_label, num_classes=num_classes)
        return hsi_batch_patch, hsi_batch_pca, hsi_batch_label

def ReadDatasets():
    class DataSets(object):
        pass
    data_sets = DataSets()
    data_sets.train = DataSet(labeled_sets)
    data_sets.valid = DataSet(valid_sets)
    data_sets.test  = DataSet(test_sets)
    data_sets.all   = DataSet(all_sets)
    return data_sets

def CalAccuracy(true_label, pred_label, class_num):
    M  = 0
    C  = np.zeros((class_num + 1, class_num + 1))
    c1 = confusion_matrix(true_label, pred_label)
    C[0:class_num, 0:class_num] = c1
    C[0:class_num, class_num]   = np.sum(c1, axis=1)
    C[class_num, 0:class_num]   = np.sum(c1, axis=0)
    N = np.sum(np.sum(c1, axis=1))
    C[class_num, class_num] = N  # all of the pixel number
    OA = np.trace(C[0:class_num, 0:class_num]) / N
    every_class = np.zeros((class_num + 3,))
    for i in range(class_num):
        acc = C[i, i] / C[i, class_num]
        M   = M + C[class_num, i] * C[i, class_num]
        every_class[i] = acc

    kappa = (N * np.trace(C[0:class_num, 0:class_num]) - M) / (N * N - M)
    AA = np.sum(every_class, axis=0) / class_num
    every_class[class_num]     = OA
    every_class[class_num + 1] = AA
    every_class[class_num + 2] = kappa
    return every_class, C

def ColorResult(each_class):
    colorbar = np.array([[0, 0, 255], [255, 0, 0], [0, 255, 0], 
                         [255, 255, 0], [0, 100, 0], [255, 0, 255], 
                         [0, 191, 255], [255, 140, 0], [255, 231, 186]])
    data = ReadDatasets()
    all_sets_index = data.all._images
    image       = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
    groundtruth = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.int64)
    for i in range(len(all_sets_index)):
        row = all_sets_index[i] // label.shape[1]
        col = all_sets_index[i] %  label.shape[1]
        for k in range(1, 10):  # num_classes=9
            if label[row, col] == k:
                groundtruth[:, row, col] = colorbar[k-1]
            if each_class[i]   == k:
                image[:, row, col] = colorbar[k-1]
    image = np.transpose(image, (1, 2, 0))
    groundtruth = np.transpose(groundtruth, (1, 2, 0))
    scipy.misc.imsave('D:/SSAN/paviac/merge/merge.jpg', image)
    scipy.misc.imsave('D:/SSAN/paviac/merge/gt.jpg', groundtruth)
    return image

# Implementation of the GAT model is left as an exercise to the user, as it involves using a library like DGL or PyTorch Geometric, 
# which is outside the scope of this code snippet.

# Usage:
# 1. Read datasets using ReadDatasets()
# 2. Train and test your GAT model on the transformed data
# 3. Calculate accuracy using CalAccuracy()
# 4. Visualize results using ColorResult()
