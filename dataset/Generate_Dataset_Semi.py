import cv2
import sys
import pickle
import random
import numpy as np

# refer : https://www.cs.toronto.edu/~kriz/cifar.html
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
def get_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def get_data_list(file_paths):
    data_list = []
    for file_path in file_paths:
        data = get_data(file_path)
        data_length = len(data[b'filenames'])

        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]

            channel_size = 32 * 32        

            r = image_data[:channel_size]
            g = image_data[channel_size : channel_size * 2]
            b = image_data[channel_size * 2 : ]

            r = r.reshape((32, 32)).astype(np.uint8)
            g = g.reshape((32, 32)).astype(np.uint8)
            b = b.reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))
            label = one_hot(label, 10)

            data_list.append([image, label])
    return data_list

def get_data_dic(file_paths):
    data_dic = {}
    for i in range(10):
        data_dic[i] = []

    for file_path in file_paths:
        data = get_data(file_path)
        data_length = len(data[b'filenames'])

        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]

            channel_size = 32 * 32        

            r = image_data[:channel_size]
            g = image_data[channel_size : channel_size * 2]
            b = image_data[channel_size * 2 : ]

            r = r.reshape((32, 32)).astype(np.uint8)
            g = g.reshape((32, 32)).astype(np.uint8)
            b = b.reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))
            data_dic[label].append(image)

    return data_dic

train_files = ['./dataset/cifar10/data_batch_{}'.format(i) for i in range(1, 5 + 1)]
test_files = ['./dataset/cifar10/test_batch']

train_data_dic = get_data_dic(train_files)
# test_data_list = get_data_list(test_files)

CLASSES = 10

for labeled in [250, 500, 1000, 2000, 4000]:
    labeled_data_list = []
    unlabeled_data_list = []

    labeled_per_class = labeled // CLASSES

    for i in range(10):
        images = train_data_dic[i]
        label = one_hot(i, 10)

        np.random.shuffle(images)
        
        for image in images[:labeled_per_class]:
            labeled_data_list.append([image, label])
            
        for image in images[labeled_per_class:]:
            unlabeled_data_list.append(image)
        
    train_dic = {
        'labeled' : labeled_data_list,
        'unlabeled' : unlabeled_data_list
    }

    print('# Labeled = {}, Labeled_per_class = {}'.format(labeled, labeled_per_class))
    print(len(labeled_data_list))
    print(len(unlabeled_data_list))

    np.save('./dataset/train_{}.npy'.format(labeled), train_dic)

# np.save('./dataset/test.npy', test_data_list)

'''
# Labeled = 250, Labeled_per_class = 25
250
49750
# Labeled = 500, Labeled_per_class = 50
500
49500
# Labeled = 1000, Labeled_per_class = 100
1000
49000
# Labeled = 2000, Labeled_per_class = 200
2000
48000
# Labeled = 4000, Labeled_per_class = 400
4000
46000
'''
