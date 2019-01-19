import numpy as np
from sklearn.model_selection import train_test_split
from model import RCGAN
import os
import argparse

FILE_NAME = 'inputs/sin_wave.npz'
# FILE_NAME = 'inputs/mnist1.npz'
SEED = 12345

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-inputs', default=FILE_NAME)
    args = parser.parse_args()
    
    # load data
    ndarr = np.load(args.inputs)
    X_train, X_eval, y_train, y_eval = train_test_split(ndarr['x'],
                                                        ndarr['y'],
                                                        test_size=0.1,
                                                        random_state=SEED)

    assert X_train.ndim == 3, 'x shape is expected 3 dims, but {} shapes'.format(
        X_train.ndim)

    print('train x shape:', X_train.shape)

    # hyper parameter for training
    args = {}
    args['seq_length'] = X_train.shape[1]
    args['input_dim'] = X_train.shape[2]
    args['latent_dim'] = 50
    args['hidden_dim'] = 100
    args['embed_dim'] = 10
    args['n_epochs'] = 30
    args['batch_size'] = 32
    args['num_classes'] = len(np.unique(y_train))
    args['save_model'] = True
    args['instance_noise'] = False
    args['dp_sgd'] = False
    args['sigma'] = 4.0
    args['l2norm_bound'] = 0.1
    args['learning_rate'] = 0.1
    args['total_examples'] = X_train.shape[0]
    
    if not os.path.isdir('models') and args['save_model']:
        os.mkdir('models')
        print('make directory for save models')

    rcgan = RCGAN(**args)

    rcgan.train(args['n_epochs'],
                X_train,
                y_train,
                X_eval,
                y_eval)


if __name__ == '__main__':
    # choose GPU devise
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    main()
