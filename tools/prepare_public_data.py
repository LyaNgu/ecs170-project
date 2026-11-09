'''
Builds MNIST / ORL / CIFAR pickle files in the course format from the public copies of the datasets:
    {'train': [{'image': ndarray, 'label': int}, ...], 'test': [...]}
Only needed if you don't have the instructor's files. Usage:
    python prepare_public_data.py <mnist_dir> <orl_dir> <cifar_dir> <out_dir>
mnist_dir : the 4 *-ubyte.gz files       orl_dir : u1..u40 folders with 1.png..10.png
cifar_dir : cifar-10-batches-py folder   ORL split: images 1-9 of each person = train, 10 = test
'''
import gzip, os, pickle, sys
import numpy as np
from PIL import Image


def mnist(d):
    def imgs(f):
        with gzip.open(os.path.join(d, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=16).reshape(-1, 28, 28)
    def labs(f):
        with gzip.open(os.path.join(d, f)) as g:
            return np.frombuffer(g.read(), np.uint8, offset=8)
    pack = lambda X, y: [{'image': a, 'label': int(b)} for a, b in zip(X, y)]
    return {'train': pack(imgs('train-images-idx3-ubyte.gz'), labs('train-labels-idx1-ubyte.gz')),
            'test': pack(imgs('t10k-images-idx3-ubyte.gz'), labs('t10k-labels-idx1-ubyte.gz'))}


def orl(d):
    train, test = [], []
    for person in range(1, 41):
        for k in range(1, 11):
            g = np.array(Image.open(os.path.join(d, 'u%d' % person, '%d.png' % k)).convert('L'))  # 112 x 92
            img = np.stack([g, g, g], axis=-1)            # stored as 3 equal channels, like the course file
            (test if k == 10 else train).append({'image': img, 'label': person})
    return {'train': train, 'test': test}


def cifar(d):
    def batch(f):
        with open(os.path.join(d, f), 'rb') as h:
            b = pickle.load(h, encoding='bytes')
        X = b[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return [{'image': a, 'label': int(l)} for a, l in zip(X, b[b'labels'])]
    train = sum([batch('data_batch_%d' % i) for i in range(1, 6)], [])
    return {'train': train, 'test': batch('test_batch')}


if __name__ == '__main__':
    mnist_dir, orl_dir, cifar_dir, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    for name, data in [('MNIST', mnist(mnist_dir)), ('ORL', orl(orl_dir)), ('CIFAR', cifar(cifar_dir))]:
        with open(os.path.join(out, name), 'wb') as f:
            pickle.dump(data, f)
        print(name, len(data['train']), len(data['test']), data['train'][0]['image'].shape)
