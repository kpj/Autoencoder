# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.decomposition import PCA

import seaborn as sns
import matplotlib.pyplot as plt

from tqdm.auto import tqdm, trange

# +
tfd = tfp.distributions

plt.set_cmap('gray')
sns.set_context('talk')

os.makedirs('images', exist_ok=True)
# -

# # Glossary

# * epoch: one cycle through training data (made up of many steps)
# * step: one gradient update per batch of data
# * learning rate: how fast to follow gradient (in gradient descent)

# # Introduction

# ## Basic operations

# +
a = tf.constant(2.)
b = tf.constant(3.)

print(tf.add(a, b))
print(tf.reduce_mean([a, b]))

# +
m1 = tf.constant([[1., 2.], [3., 4.]])
m2 = tf.constant([[5., 6.], [7., 8.]])

tf.matmul(m1, m2)


# -

# # Basic Autoencoder

# ## Loss function

# $$
# \frac{1}{N} \sum_{i=0}^N (\hat{x}_i - x_i)^2
# $$

# ## Design model

# +
class Encoder(tf.keras.layers.Layer):
    """Convert input to low-dimensional representation."""

    def __init__(self, latent_dim, original_dim):
        super(Encoder, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(*original_dim, 1)),

            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=2*latent_dim)
        ])

    def call(self, x):
        return self.network(x)


class Decoder(tf.keras.layers.Layer):
    """Reconstruct input from low-dimensional representation."""

    def __init__(self, latent_dim, original_dim):
        super(Decoder, self).__init__()
        self.network = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),

            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=(2, 2), padding='SAME', activation=tf.nn.relu),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=(2, 2), padding='SAME', activation=tf.nn.relu),

            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding='SAME'),
        ])

    def call(self, z):
        return self.network(z)


class Autoencoder(tf.keras.Model):
    """Connect all components to single model."""

    def __init__(self, latent_dim, original_dim):
        """Initialize everything."""
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        # setup architecture
        self._encoder = Encoder(
            latent_dim=self.latent_dim, original_dim=original_dim)
        self._decoder = Decoder(
            latent_dim=self.latent_dim, original_dim=original_dim)

        # helpful stuff
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
    
    def compute_loss(self, data):
        return tf.reduce_mean(
            tf.square(
                tf.subtract(
                    self(data),
                    data
                )
            )
        )

    @tf.function
    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(data)

            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_loss(loss)

    def save(self, fname):
        """Save model.
            https://www.tensorflow.org/alpha/guide/keras/saving_and_serializing#saving_subclassed_models
        """
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        self.save_weights(fname, save_format='tf')

    @classmethod
    def load_from_file(cls, fname):
        model = cls(latent_dim, original_dim)  # TODO: load parameters from file

        # train model briefly to infer architecture
        # TODO

        model.load_weights(fname)
        return model


# -

# ## Run model

# ### Parameters

np.random.seed(1)
tf.random.set_seed(1)

batch_size = 128
epochs = 20
learning_rate = 1e-3
latent_dim = 10
original_dim = (28, 28)

# ### Load data

# +
(training_features, training_labels), (test_features, _) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)

# flatten 2D images into 1D
training_features = training_features.reshape(training_features.shape[0], *original_dim, 1).astype(np.float32)

# prepare dataset
training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)
training_dataset
# -

# ### Train model

autoencoder = Autoencoder(latent_dim=latent_dim, original_dim=original_dim)

opt = tf.optimizers.Adam(learning_rate=learning_rate)

loss_list = []
for epoch in trange(epochs, desc='Epochs'):
    for step, batch_features in enumerate(training_dataset):
        autoencoder.train_step(batch_features, opt)

    loss_list.append(autoencoder.train_loss.result().numpy())

autoencoder.save('models/autoencoder')

# autoencoder = Autoencoder.load_from_file('models/autoencoder')

# ## Analysis

# ### Loss development

# +
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.savefig('images/loss_function.pdf')
# -

# ### Fun in latent space

inp = np.random.normal(size=64)
plt.imshow(autoencoder.decoder([inp]).numpy()[0].reshape(28, 28))

# ### Check specific examples

original = tf.reshape(
    batch_features, (batch_features.shape[0], 28, 28))
reconstructed = tf.reshape(
    autoencoder(tf.constant(batch_features)),
    (batch_features.shape[0], 28, 28))

# +
N = 16
fig, ax_list = plt.subplots(nrows=4, ncols=4, figsize=(8, 4))

for i, ax in enumerate(ax_list.ravel()):
    idx = np.random.randint(0, original.shape[0])

    img_orig = original.numpy()[idx]
    img_recon = reconstructed.numpy()[idx]

    img_concat = np.concatenate([img_orig, img_recon], axis=1)

    ax.imshow(img_concat)
    ax.axis('off')


# -

# ### PCA

def do_PCA(X, ndim=2):
    pca = PCA(n_components=ndim)
    pca.fit(X)
    X_trans = pca.transform(X)
    print(pca.explained_variance_ratio_)
    return pd.DataFrame(X_trans, columns=[f'PC_{i}' for i in range(ndim)])


# +
latent_features = autoencoder.encode(training_features)
latent_features_1d = np.concatenate(latent_features, axis=1)

df_latent = do_PCA(latent_features_1d)
df_latent['label'] = training_labels
df_latent['space'] = 'latent'

# +
data_train_1d = training_features.reshape(
    training_features.shape[0],
    training_features.shape[1]*training_features.shape[2])

df_original = do_PCA(data_train_1d)
df_original['label'] = training_labels
df_original['space'] = 'original'
# -

df_pca = pd.concat([df_original, df_latent])

g = sns.FacetGrid(df_pca, col='space', hue='label', height=8, legend_out=True)
g.map_dataframe(sns.scatterplot, x='PC_0', y='PC_1', rasterized=True)
g.add_legend()
g.savefig('images/pca.pdf')
