import numpy as np

import AutoAugment.policies as found_policies
import AutoAugment.augmentation_transforms as transform

good_policies = found_policies.good_policies()

def AutoAugment(image, normalize = False):
    random_policy = good_policies[np.random.choice(len(good_policies))]

    if not normalize:
        image = image.astype(np.float32) / 255.
        image = (image - transform.MEANS) / transform.STDS

    image = transform.apply_policy(random_policy, image)
    image = transform.zero_pad_and_crop(image, 4)
    image = transform.random_flip(image)
    image = transform.cutout_numpy(image)
    
    if not normalize:
        image = ((image * transform.STDS) + transform.MEANS) * 255.

    return image