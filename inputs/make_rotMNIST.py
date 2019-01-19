"""
rotMNISTからGAN学習用のデータセットを作成する
"""
import numpy as np
import json

FILNAME = 'mnist1.npz'
SAMPLES = 3000
MAX_SEQ = 10
INPUT_DIM = 784


def seq_padding(data, max_seq, pad_value=1):
    seq, dim = data.shape
    if max_seq > seq:
        diff_seq = max_seq - seq
        pad_seq = np.zeros((diff_seq, dim)) + pad_value
        pad_data = np.concatenate([data, pad_seq], axis=0)
        return pad_data
    else:
        return data[:max_seq]


def main():
    jfile = open('rotMNIST/rotMNIST.json', 'r')
    json_dic = json.load(jfile)

    data_arr = np.empty((SAMPLES, MAX_SEQ, INPUT_DIM))

    label_list = []
    for idx in range(SAMPLES):
        if idx % 100 == 0: print('loading ... {}'.format(idx))
        idx_str = '{0:06d}'.format(idx)
        npy_file = json_dic['rotMNIST/data/' + idx_str]['data'][0]
        data_arr[idx] = seq_padding(np.load(npy_file), MAX_SEQ)
        label = json_dic['rotMNIST/data/' + idx_str]['label']
        label_list.append(label)
    label_arr = np.array(label_list)
    print('load complete !\n')
    print('data shape .. ', data_arr.shape)
    print('label shape ..', label_arr.shape)
    print('label classes ..', np.unique(label_arr))

    assert np.isnan(
        data_arr).any() == False, 'inputs data have null value. RCGAN do not allow null value'

    np.savez(FILNAME, x=data_arr, y=np.array(label_list))

    print('data is saved .. {}'.format(FILNAME))


if __name__ == '__main__':

    main()
