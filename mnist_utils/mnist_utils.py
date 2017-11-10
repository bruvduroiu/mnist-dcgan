import struct
import os
import numpy as np

def read(dataset='training', path='.'):

    if dataset is 'training':
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is 'testing':
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')

    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack('>II', flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.uint8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack('>IIII', fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    return (lbl, img)


def show(image):
    from matplotlib import pyplot as plt
    import matplotlib as mpl

    fig = plt.figure()
    ax = fig.add_subplot(111)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()