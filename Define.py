
IMAGE_SIZE = 32
IMAGE_CHANNEL = 3

'''
0 : airplane
1 : automobile
2 : bird
3 : cat
4 : deer
5 : dog
6 : frog
7 : horse
8 : ship
9 : truck
'''
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASSES = len(CLASS_NAMES)

EMA_DECAY = 0.999
INIT_LEARNING_RATE = 0.03
WEIGHT_DECAY = 0.0005

WARMUP_ITERATION = 20000
MAX_ITERATION = 100000
DECAY_ITERATIONS = [50000, 75000]

NUM_THREADS = 5

SU_RATIOS = 1
BATCH_SIZE = 32

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
