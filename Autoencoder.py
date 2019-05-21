# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.1.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
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

def loss_func(model, original):
    return tf.reduce_mean(
        tf.square(
            tf.subtract(
                model(original),
                original
            )
        )
    )


# ## Design model

# +
class Encoder(tf.keras.layers.Layer):
    """Convert input to low-dimensional representation."""

    def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(
            units=intermediate_dim, activation=tf.nn.relu)

    def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)


class Decoder(tf.keras.layers.Layer):
    """Reconstruct input from low-dimensional representation."""

    def __init__(self, intermediate_dim, original_dim):
        super(Decoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(
            units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(
            units=original_dim, activation=tf.nn.relu)

    def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
    """Connect all components to single model."""

    def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        
        # setup architecture
        self.encoder = Encoder(
            intermediate_dim=intermediate_dim)
        self.decoder = Decoder(
            intermediate_dim=intermediate_dim, original_dim=original_dim)
        
        # helpful stuff
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed
    
    @tf.function
    def train_step(self, data, optimizer):
        with tf.GradientTape() as tape:
            loss = loss_func(self, data)
            
            gradients = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.trainable_variables))
            
        self.train_loss(loss)


# -

# ## Run model

# ### Parameters

np.random.seed(1)
tf.random.set_seed(1)

batch_size = 128
epochs = 20
learning_rate = 1e-3
intermediate_dim = 64
original_dim = 784

# ### Load data

# +
(training_features, training_labels), (test_features, _) = tf.keras.datasets.mnist.load_data()
training_features = training_features / np.max(training_features)

# flatten 2D images into 1D
training_features = training_features.reshape(
    training_features.shape[0],
    training_features.shape[1] * training_features.shape[2]).astype(np.float32)

training_dataset = tf.data.Dataset.from_tensor_slices(training_features).batch(batch_size)
training_dataset
# -

# ### Train model

autoencoder = Autoencoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

opt = tf.optimizers.Adam(learning_rate=learning_rate)

loss_list = []
for epoch in trange(epochs, desc='Epochs'):
    for step, batch_features in enumerate(training_dataset):
        autoencoder.train_step(batch_features, opt)
        
    loss_list.append(autoencoder.train_loss.result().numpy())

# ## Analysis

# ### Loss development

plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')

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
N = 2
fig, ax = plt.subplots(nrows=N, ncols=2, figsize=(10, N*5))

for i, ap in enumerate(ax):
    ap[0].imshow(original.numpy()[i])
    ap[1].imshow(reconstructed.numpy()[i])

    if i == 0:
        ap[0].set_title('Input')
        ap[1].set_title('Output')

    for a in ap:
        a.set_xticks([])
        a.set_yticks([])


# -

# ### PCA

def do_PCA(X, ndim=2):
    pca = PCA(n_components=ndim)
    pca.fit(X)
    X_trans = pca.transform(X)
    print(pca.explained_variance_ratio_)
    return pd.DataFrame(X_trans, columns=[f'PC_{i}' for i in range(ndim)])


# +
latent_features = autoencoder.encoder(training_features)

df_latent = do_PCA(latent_features)
df_latent['label'] = training_labels
df_latent['space'] = 'latent'
# -

df_original = do_PCA(training_features)
df_original['label'] = training_labels
df_original['space'] = 'original'

df_pca = pd.concat([df_original, df_latent])

g = sns.FacetGrid(df_pca, col='space', hue='label', height=8, legend_out=True)
g.map_dataframe(sns.scatterplot, x='PC_0', y='PC_1')
g.add_legend()
