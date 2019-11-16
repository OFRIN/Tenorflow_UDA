
import cv2
import time

import numpy as np
import tensorflow as tf

import AutoAugment.AutoAugment as augment

'''
all_data = all_data / 255.0
mean = augmentation_transforms.MEANS
std = augmentation_transforms.STDS
all_data = (all_data - mean) / std
'''

train_data_dic = np.load('./dataset/train_{}.npy'.format(4000), allow_pickle = True)

labeled_data_list = train_data_dic.item().get('labeled')
unlabeled_data_list = train_data_dic.item().get('unlabeled')

# good_policies = found_policies.good_policies()

# mean = transform.MEANS
# std = transform.STDS
# print(mean, std)

# epoch_policy = good_policies[np.random.choice(len(good_policies))]

# for label_data in labeled_data_list:
#     image, label = label_data

#     augment_image = image.astype(np.float32)
#     augment_image /= 255.0
#     augment_image = (augment_image - mean) / std

#     final_img = transform.apply_policy(epoch_policy, augment_image)
#     final_img = transform.random_flip(transform.zero_pad_and_crop(final_img, 4))

#     # Apply cutout
#     final_img = transform.cutout_numpy(final_img)
#     final_img = ((final_img * std) + mean) * 255.

#     image = cv2.resize(image.astype(np.uint8), (224, 224))
#     augment_image = cv2.resize(final_img.astype(np.uint8), (224, 224))

#     cv2.imshow('show', image)
#     cv2.imshow('augment_show', augment_image)
#     cv2.waitKey(0)

for label_data in labeled_data_list:
    image, label = label_data
    
    st_time = time.time()

    augment_image = augment.AutoAugment(image.astype(np.float32), normalize = False)

    end_time = time.time()
    print(int((end_time - st_time) * 1000))

    augment_image = cv2.resize(augment_image.astype(np.uint8), (224, 224))
    cv2.imshow('show', augment_image)
    cv2.waitKey(0)
