import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.python.client import device_lib


# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def gpu_is_available():
    local_device_protos = device_lib.list_local_devices()
    available_gpu = [x.name for x in local_device_protos if
                     x.device_type == 'GPU']
    return True if available_gpu else False


def draw_sin_wave(sin_wave, labels, epoch):
    num_samples = sin_wave.shape[0]
    sin_wave = np.squeeze(sin_wave)
    plt.figure(figsize=(12, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.plot(sin_wave[i])
        if labels is not None:
            plt.title(labels[i])
        plt.ylim(-1.0, 1.0)
        plt.axis('off')
    plt.savefig('plot/sin_wave_samples_epoch{}.png'.format(epoch))
    plt.close()


def draw_mnist(img, labels, epoch):
    num_samples = img.shape[0]
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img[i], cmap='gray')
        plt.title(labels[i])
        plt.axis('off')
    plt.savefig('plot/MNIST_samples_epoch{}.png'.format(epoch))
    plt.close()


def draw_rotmnist(img, labels, epoch):
    plt.figure(figsize=(12, 8))

    show_num = 1
    for i in range(10):
        for j in range(5):
            plt.subplot(5, 10, show_num)
            plt.imshow(img[i][j].reshape(28, 28), cmap='gray')
            plt.title(labels[i])
            plt.axis('off')
            show_num += 1
    plt.savefig('plot/rotMNIST_samples_epoch{}.png'.format(epoch))
    plt.close()
    
    
def data_shuffle(x, y):
    # shuffle x and y
    rnd = np.random.randint(999)
    for l in [x, y]:
        np.random.seed(rnd)
        np.random.shuffle(l)


def DPSGD(sigma, l2norm_bound, learning_rate, total_examples):
    import tensorflow as tf
    from differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
    from differential_privacy.dp_sgd.dp_optimizer import sanitizer
    from differential_privacy.privacy_accountant.tf import accountant

    eps = tf.placeholder(tf.float32)
    delta = tf.placeholder(tf.float32)

    priv_accountant = accountant.GaussianMomentsAccountant(total_examples)
    clip = True
    batches_per_lot = 1

    gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
                    priv_accountant,
                    [l2norm_bound, clip])

    return dp_optimizer.DPGradientDescentOptimizer(learning_rate,
                                                  [eps, delta],
                                                  sanitizer=gaussian_sanitizer,
                                                  sigma=sigma,
                                                  batches_per_lot=batches_per_lot)