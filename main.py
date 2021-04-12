import tensorflow as tf
import matplotlib.pyplot as plt
import numpy
from tensorflow.keras.datasets import fashion_mnist
# We don't need y_train and y_test
(x_train, _), (x_test, _) = fashion_mnist.load_data()
print('Max value in the x_train is', x_train[0].max())
print('Min value in the x_train is', x_train[0].min())


fig, axs = plt.subplots(5, 10)
fig.tight_layout(pad=-1)
plt.gray()
a = 0
for i in range(5):
  for j in range(10):
    axs[i, j].imshow(tf.squeeze(x_test[a]))
    axs[i, j].xaxis.set_visible(False)
    axs[i, j].yaxis.set_visible(False)
    a = a + 1
x_train  =  x_train.astype ( 'float32' )/255.
x_test  =  x_test.astype ( 'float32' )/255.
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]
print(x_train.shape)
print(x_test.shape)
noise_factor = 0.4
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)

n = 5
plt.figure(figsize=(20, 8))
plt.gray()
for i in range(n):
  ax = plt.subplot(2, n, i + 1)
  plt.title("original", size=20)
  plt.imshow(tf.squeeze(x_test[i]))
  plt.gray()
  bx = plt.subplot(2, n, n+ i + 1)
  plt.title("original + noise", size=20)
  plt.imshow(tf.squeeze(x_test_noisy[i]))
plt.show()
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
import os.path



if os.path.exists('model')==False or input('Загрузить прошлую модель?(Y/N): ')== 'N':
    class NoiseReducer(tf.keras.Model):
        def __init__(self):
            super(NoiseReducer, self).__init__()

            self.encoder = tf.keras.Sequential([
                Input(shape=(28, 28, 1)),
                Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
                Conv2D(8, (3, 3), activation='relu', padding='same', strides=2),])

            self.decoder = tf.keras.Sequential([
                Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
                Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
                Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    autoencoder = NoiseReducer()
    autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    autoencoder.fit(x_train_noisy,
                     x_train,
                     epochs=1,
                     shuffle=True,
                     validation_data=(x_test_noisy, x_test))
    encoded_imgs=autoencoder.encoder(x_test_noisy).numpy()
    decoded_imgs=autoencoder.decoder(encoded_imgs)
else:
    autoencoder = tf.keras.models.load_model("model")
    encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
    decoded_imgs = autoencoder.decoder(encoded_imgs)
n = 10
plt.figure(figsize=(20, 7))
plt.gray()
for i in range(n):
  # display original + noise
  bx = plt.subplot(3, n, i + 1)
  plt.title("original + noise")
  plt.imshow(tf.squeeze(x_test_noisy[i]))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  cx = plt.subplot(3, n, i + n + 1)
  plt.title("reconstructed")
  plt.imshow(tf.squeeze(decoded_imgs[i]))
  bx.get_xaxis().set_visible(False)
  bx.get_yaxis().set_visible(False)

  # display original
  ax = plt.subplot(3, n, i + 2*n + 1)
  plt.title("original")
  plt.imshow(tf.squeeze(x_test[i]))
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

autoencoder.save('model')

plt.show()