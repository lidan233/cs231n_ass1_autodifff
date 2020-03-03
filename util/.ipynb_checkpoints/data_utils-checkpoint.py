import numpy as np
import os
import scipy.misc as imread
import pickle as pc

def load_cifar_batch(filename):
    with open(filename,'rb') as f:
        datadict = pc.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000,3,32,32).transpose(0,2,3,1).astype("float")
        Y = np.array(Y)
        return X,Y

def load_CIFAR10(ROOT):
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT,"data_batch_%d"%(b,))
        X,Y = load_cifar_batch(f)
        xs.append(X)
        ys.append(Y)

    # concat to multi axis
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)

    del X,Y
    Xte,Yte = load_cifar_batch(os.path.join(ROOT,'test_batch'))
    return Xtr,Ytr,Xte,Yte

def load_tiny_imagenet(path,dtype=np.float32):

    # wnid name label is a tuple relation

    # First load wnids
    with open(os.path.join(path, 'wnids.txt'), 'r') as f:
        wnids = [x.strip() for x in f]

    #every label has a number
    wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

    # Use words.txt to get names for each class
    with open(os.path.join(path,'words.txt'),'r+') as f:
        wnid_to_words = dict(line.split('\t') for line in f)

        # iteritems change to items
        for wnid,words in wnid_to_words.items():
            wnid_to_words[wnid] = [w.strip for w in words.split(',')]
    class_names = [wnid_to_words[wnid] for wnid in wnids]


    # get train
    Xtr = []
    Ytr = []
    for i ,wnid in enumerate(wnids):
        if (i + 1) % 20 == 0:
            #
            print('loading training data for synset %d / %d' % (i + 1, len(wnids)))
        # To figure out the filenames we need to open the boxes file
        boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
        with open(boxes_file, 'r') as f:
            filenames = [x.split('\t')[0] for x in f]
        num_images = len(filenames)

        X_train_block = np.zeros((num_images, 3, 64, 64), dtype=dtype)
        y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
        for j, img_file in enumerate(filenames):
            img_file = os.path.join(path, 'train', wnid, 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                ## grayscale file
                img.shape = (64, 64, 1)
            # (1,64,64)
            X_train_block[j] = img.transpose(2, 0, 1)
        Xtr.append(X_train_block)
        Ytr.append(y_train_block)


    Xtr = np.concatenate(Xtr,axis=0)
    Ytr = np.concatenate(Ytr,axis=0)

    # Next load validation data
    with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
        img_files = []
        val_wnids = []
        for line in f:
            img_file, wnid = line.split('\t')[:2]
            img_files.append(img_file)
            val_wnids.append(wnid)
        num_val = len(img_files)
        y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
        X_val = np.zeros((num_val, 3, 64, 64), dtype=dtype)
        for i, img_file in enumerate(img_files):
            img_file = os.path.join(path, 'val', 'images', img_file)
            img = imread(img_file)
            if img.ndim == 2:
                img.shape = (64, 64, 1)
            X_val[i] = img.transpose(2, 0, 1)

    # Next load test images
    # Students won't have test labels, so we need to iterate over files in the
    # images directory.
    img_files = os.listdir(os.path.join(path, 'test', 'images'))
    X_test = np.zeros((len(img_files), 3, 64, 64), dtype=dtype)
    for i, img_file in enumerate(img_files):
        img_file = os.path.join(path, 'test', 'images', img_file)
        img = imread(img_file)
        if img.ndim == 2:
            img.shape = (64, 64, 1)
        X_test[i] = img.transpose(2, 0, 1)

    y_test = None
    y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
    if os.path.isfile(y_test_file):
        with open(y_test_file, 'r') as f:
            img_file_to_wnid = {}
            for line in f:
                line = line.split('\t')
                img_file_to_wnid[line[0]] = line[1]
        y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
        y_test = np.array(y_test)

    return class_names, Xtr, Ytr, X_val, y_val, X_test, y_test



def load_models(models_dir):
    models = {}
    for models_file in os.listdir(models_dir):
        with open(os.path.join(models_dir,models_file),'rb') as f:
            try:
                models[models_file] = pc.load(f)['model']
            except pc.UnpicklingError:
                continue
    return models 
