# # Unsupervised Data Augmentation

### # Summary
- To learn many images and few labels is very difficult. Many machine learning engineers know collecting label is expensive for many learning tasks because collecting labels need expert knowledge. In contrast, To collect unlabeled images is more easy and fast than labeling. (Generally dataset mean is pair of image and label.) UDA needs many unlabeled images and few labeled images. Therefore, 기존보다 훨씬 적은 시간으로 학습이 가능하기 때문에 많은 머신러닝 개발자들은 손쉽게 프로젝트를 진행할 수 있다. 모든 Semi Supervised Learning의 단점은 결국 100% 레이블링된 데이터셋의 근접할 뿐 결국 100% 레이블링한 데이터셋으로 학습한 모델의 성능이 가장 좋다는 점이다.

- UDA has two main contributions. 
  1. UDA applies to consistency training which induces unlabeled image and unlabeled image with RandAugment equally.
  2. To block overfitting about few labeled images use training signal annealing, confidence mask, and sharpening.

- Before explaining about UDA, I define three keywords.
  First, Consistency training reduces uncertainty of different images.
  Second, 
  Third,

- UDA outperforms previous state-of-the-art methods from many vision tasks. For example, UDA achieved test error 5.29% from CIFAR-10 with 4,000 labels. (4,000 labels is 10% on CIFAR-10)

