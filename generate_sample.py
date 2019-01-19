"""
generate samples from learned model
"""
import matplotlib
matplotlib.use('Agg')
from keras.models import model_from_json
from mmd import tf_initialize, sigma_optimization
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import utils

####### edit this variable depending on the situation ##########
# file path
INPUT_FILE_NAME = 'inputs/sin_wave.npz'  # real data for mmd
OUTPUT_FILE_NAME = 'outputs/samples.npz'  # generated samples
PLOT_FILE_NAME = 'plot/generated_samples.png'
G_MODEL_FILE_NAME = 'models/generator_model.json'  # generater model
G_WEIGHT_FILE_NAME = 'models/generator_weight.h5'  # generater weight
D_MODEL_FILE_NAME = 'models/discriminator_model.json'  # discriminator model
D_WEIGHT_FILE_NAME = 'models/discriminator_weight.h5'  # discriminator weight
# hyper parameter for generating sampels
batch_size = 32  # batch size (adjust this variable to your model) 
input_dim = 1  # input dimension (adjust this variable to your model) 
seq_length = 30  # sequence length (adjust this variable to your model) 
latent_dim = 50  # latent dimension for gan (adjust this variable to your model) 
num_classes = 3  # a number of classes (adjust this variable to your model) 
num_samples = 500  # a number of generated samples
eval_mmd = True  # evaluation of mmd score
plot = False  # cannot plot without sin wave
show_samples = 15  # a number of plotting samples
################################################################

class MMD():
    def __init__(self,
                 seq_length,
                 input_dim,
                 latent_dim):
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.sigma = 0
        self.dic_tf_sigma = None
        self.that = 0
        self.sess = None

    def set_sigma(self, x_eval):
        self.sigma, self.dic_tf_sigma, self.that, self.sess = tf_initialize(
            x_eval,
            self.seq_length,
            self.input_dim)

    def calc_mmd(self, eval_real, eval_gen):
        eval_real = np.float32(eval_real)
        eval_gen = np.float32(eval_gen)
        # get MMD
        mmd2, that_np = sigma_optimization(eval_real,
                                           eval_gen,
                                           self.sigma,
                                           self.dic_tf_sigma,
                                           self.that,
                                           self.sess)
        return mmd2, that_np



def generate_samples():
    gen_x = []
    gen_y = []
    gen_iter = (num_samples // batch_size) + 1
    for i in range(0, gen_iter):
        noise = np.random.normal(0, 1, (batch_size, seq_length, latent_dim))
        sample_c = np.random.randint(0, num_classes, batch_size)
        gen_x.extend(generator.predict([noise, sample_c]))
        gen_y.extend(sample_c)
    gen_x_arr = np.array(gen_x)
    gen_y_arr = np.array(gen_y)
    return gen_x_arr[:num_samples], gen_y_arr[:num_samples]


def plotting():
    idx = np.random.randint(0, gen_x.shape[0], show_samples)
    plt.figure(figsize=(20, 8))
    # show sin wave
    for i in range(show_samples):
        plt.subplot(3, show_samples / 3, i + 1)
        plt.plot(gen_x[idx[i]].squeeze())
        plt.ylim(-0.8, 0.8)
        plt.title(gen_y[idx[i]])
        plt.axis('off')
    plt.savefig(PLOT_FILE_NAME)
    plt.close()


if __name__ == '__main__':
    
    # choose GPU devise
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
    # get argument if exit
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=num_samples)
    args = parser.parse_args()
    num_samples = args.n 
    
    # initialize MMD Class
    my_mmd = MMD(seq_length, input_dim, latent_dim)

    # load learned model
    generator = model_from_json(open(G_MODEL_FILE_NAME).read())
    generator.load_weights(G_WEIGHT_FILE_NAME)
    discriminator = model_from_json(open(D_MODEL_FILE_NAME).read())
    discriminator.load_weights(D_WEIGHT_FILE_NAME)

    # generate samples
    gen_x, gen_y = generate_samples()
    
    if eval_mmd:
        ndarr = np.load(INPUT_FILE_NAME)
        real_x = ndarr['x']
        _ = ndarr['y']
        eval_x = real_x[np.random.choice(real_x.shape[0], num_samples, replace=False)]
        test_x = real_x[np.random.choice(real_x.shape[0], num_samples, replace=False)]
        # eval_size = int(real_x.shape[0] * 0.1)
        my_mmd.set_sigma(eval_x)
        mmd2, that_np = my_mmd.calc_mmd(test_x, gen_x)
        print('[mmd2: {:.3f}]'.format(mmd2))
    
    if plot:
        if input_dim > 1: 
            'input dimension is invalid for plotting'
        else:
            plotting()
        
    np.savez(OUTPUT_FILE_NAME, x=gen_x, y=gen_y)
    
    print('generete {} samples successfully !'.format(num_samples))
    print('samples are saved to {}'.format(OUTPUT_FILE_NAME))
        
    