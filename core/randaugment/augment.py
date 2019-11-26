import random
import numpy as np

import core.randaugment.policies as found_policies
import core.randaugment.augmentation_transforms as transform

class RandAugment:
    def __init__(self):
        self.mean, self.std = transform.get_mean_and_std()
        self.polices = found_policies.randaug_policies()

    def __call__(self, image):
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        
        chosen_policy = random.choice(self.polices)
        aug_image = transform.apply_policy(chosen_policy, image)
        aug_image = transform.cutout_numpy(aug_image)

        return aug_image
