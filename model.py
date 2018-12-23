import numpy as np
from keras.layers import Input, Dense, CuDNNLSTM
from keras.layers import concatenate, Flatten, Embedding, RepeatVector
from keras.layers.recurrent import LSTM
from keras.models import Sequential, Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.layers.wrappers import TimeDistributed
from keras import backend as K
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from mmd import tf_initialize, sigma_optimization
import utils


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



class RCGAN():
    def __init__(self, **kwargs):

        self.input_dim = kwargs["input_dim"]
        self.seq_length = kwargs["seq_length"]
        self.latent_dim = kwargs["latent_dim"]
        self.hidden_dim = kwargs["hidden_dim"]
        self.embed_dim = kwargs["embed_dim"]
        self.batch_size = kwargs["batch_size"]
        self.num_classes = kwargs["num_classes"]
        self.save_model = kwargs["save_model"]
        self.instance_noise = kwargs["instance_noise"]
        self.dp_sgd = kwargs['dp_sgd']
        self.sigma = kwargs['sigma']
        self.l2norm_bound = kwargs['l2norm_bound']
        self.learning_rate = kwargs['learning_rate']
        self.total_examples = kwargs['total_examples']

        # get available GPU
        self.use_gpu = utils.gpu_is_available()

        # initialize MMD Class
        self.my_mmd = MMD(self.seq_length, self.input_dim, self.latent_dim)

        # model instantiation
        self.discriminator = self.build_discriminator()
        self.generator = self.build_generator()

        # define input tenor shape
        # we define batch size here for DP-SGD
        x = Input(
            batch_shape=(self.batch_size, self.seq_length, self.input_dim))
        z = Input(
            batch_shape=(self.batch_size, self.seq_length, self.latent_dim))
        c = Input(batch_shape=(self.batch_size, 1), dtype='int32')

        self.set_trainable(self.generator, trainable=False)

        # discriminator takes real x and gererated gx
        d_logit_real = self.discriminator([x, c])
        gx = self.generator([z, c])
        d_logit_fake = self.discriminator([gx, c])

        # get loss function
        d_loss, g_loss = self.gan_loss(d_logit_real, d_logit_fake)

        # define optimizer
        if self.dp_sgd:
            print('Using differentially private SGD to train discriminator!')
            d_optim = utils.DPSGD(self.sigma, self.l2norm_bound, self.learning_rate,
                                  self.total_examples)
        else:
            d_optim = SGD(self.learning_rate)
            
        g_optim = Adam()

        # build trainable discriminator model
        self.D_model = Model([x, z, c], [d_logit_real, d_logit_fake])
        self.D_model.add_loss(d_loss)
        self.D_model.compile(optimizer=d_optim, loss=None)

        # freeze discriminator parameter when training discriminator
        self.set_trainable(self.generator, trainable=True)
        self.set_trainable(self.discriminator, trainable=False)

        # build trainable generator model
        self.G_model = Model([z, c], d_logit_fake)
        self.G_model.add_loss(g_loss)
        self.G_model.compile(optimizer=g_optim, loss=None)

    def gan_loss(self, d_logit_real, d_logit_fake):
        """
        define loss function
        """

        d_loss_real = K.mean(K.binary_crossentropy(output=d_logit_real,
                                                   target=K.ones_like(
                                                       d_logit_real),
                                                   from_logits=True))
        d_loss_fake = K.mean(K.binary_crossentropy(output=d_logit_fake,
                                                   target=K.zeros_like(
                                                       d_logit_fake),
                                                   from_logits=True))

        d_loss = d_loss_real + d_loss_fake

        g_loss = K.mean(K.binary_crossentropy(output=d_logit_fake,
                                              target=K.ones_like(d_logit_fake),
                                              from_logits=True))

        return d_loss, g_loss

    def build_generator(self):

        # define sequential model
        model = Sequential()

        if self.use_gpu:
            model.add(CuDNNLSTM(units=self.hidden_dim,
                                return_sequences=True))
        else:
            model.add(LSTM(units=self.hidden_dim,
                                return_sequences=True))

        model.add(TimeDistributed(Dense(self.input_dim, activation='tanh')))

        # define tenor variable
        z = Input(
            batch_shape=(self.batch_size, self.seq_length, self.latent_dim))
        c = Input(batch_shape=(self.batch_size, 1), dtype='int32')
        c_emb = Flatten()(Embedding(self.num_classes, self.embed_dim)(c))
        c_emb = RepeatVector(self.seq_length)(c_emb)
        # inputs = multiply([z, c_emb])
        inputs = concatenate([z, c_emb], axis=-1)

        # define generator output
        gx = model(inputs)

        return Model([z, c], gx)

    def build_discriminator(self):

        # define sequential model
        model = Sequential()

        if self.use_gpu:
            model.add(CuDNNLSTM(units=self.hidden_dim,
                                return_sequences=True))
        else:
            model.add(LSTM(units=self.hidden_dim,
                                return_sequences=True))
        # model.add(TimeDistributed(Dense(1, activation='sigmoid')))
        # pass logit value to loss function
        model.add(TimeDistributed(Dense(1)))

        # define tenor variable
        x = Input(batch_shape=(self.batch_size, self.seq_length, 1))
        c = Input(batch_shape=(self.batch_size, 1), dtype='int32')
        c_emb = Flatten()(Embedding(self.num_classes, self.embed_dim)(c))
        c_emb = RepeatVector(self.seq_length)(c_emb)

        # inputs = multiply([x, c_emb])
        inputs = concatenate([x, c_emb], axis=-1)

        # define discriminator output
        validity = model(inputs)

        return Model([x, c], validity)

    def set_trainable(self, model, trainable=False):
        model.trainable = trainable
        for layer in model.layers:
            layer.trainable = trainable

    def train(self, n_epochs, X_train, y_train, X_eval, y_eval):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        set_session(sess)

        eval_iter = X_eval.shape[0] // self.batch_size
        # eval_iter = 5
        self.my_mmd.set_sigma(X_eval[:eval_iter * self.batch_size])
        
        best_mmd2 = 999

        for epoch in range(n_epochs):

            utils.data_shuffle(X_train, y_train)

            for i in range(int(X_train.shape[0] / self.batch_size)):
                tr_x = X_train[i * self.batch_size: (i + 1) * self.batch_size]
                tr_y = y_train[i * self.batch_size: (i + 1) * self.batch_size]

                if self.instance_noise:
                    i_noise = np.random.normal(0, 0.01, (
                        self.batch_size, self.seq_length, self.input_dim))
                    tr_x += i_noise

                noise = np.random.normal(0, 1, (
                    self.batch_size, self.seq_length, self.latent_dim))

                d_loss_curr = self.D_model.train_on_batch([tr_x, noise, tr_y],
                                                          None)
                g_loss_curr = self.G_model.train_on_batch([noise, tr_y], None)

            if (epoch + 1) % 5 == 0:

                # prepare data for maximum mean discrepancy
                np.random.shuffle(X_eval)

                eval_x = []
                eval_gx = []
                for j in range(eval_iter):
                    eval_x.extend(
                        X_eval[j * self.batch_size: (j + 1) * self.batch_size])

                    noise = np.random.normal(0, 1, (
                        self.batch_size, self.seq_length, self.latent_dim))
                    sample_c = np.random.randint(0, self.num_classes,
                                                 self.batch_size)
                    eval_gx.extend(self.generator.predict([noise, sample_c]))

                mmd2, that_np = self.my_mmd.calc_mmd(eval_x, eval_gx)

                # Plot the progress
                print (
                    "epoch {} [D loss: {:.3f}] [G loss: {:.3f}] [mmd2: {:.3f}]".
                        format(epoch + 1,
                               np.mean(d_loss_curr),
                               np.mean(g_loss_curr),
                               mmd2))
                
                # save model and generate data based on current mmd2 score
                if (epoch + 1) >= 10 and best_mmd2 - mmd2 > 0.005:
                    if self.save_model:
                        model_json_str = self.generator.to_json()
                        open('models/' + '_generator_model.json', 'w') \
                            .write(model_json_str)
                        self.generator.save_weights(
                            'models/' + 'generator_weight.h5')
                        model_json_str = self.discriminator.to_json()
                        open('models/' + 'discriminator_model.json', 'w') \
                            .write(model_json_str)
                        self.discriminator.save_weights(
                            'models/' + 'discriminator_weight.h5')
                        print('best model is saved !!')

                        best_mmd2 = mmd2
                # Plot generated samples from current generator
                # sin_plot(sample_gx[:8], sample_c[:8])
