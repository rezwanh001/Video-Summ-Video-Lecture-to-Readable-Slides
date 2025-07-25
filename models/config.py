"""
@author: Md. Rezwanul Haque

"""

class CONFIG:
    # ============ TRAIN CONFIG ==============
    # Dataset options
    DATASET = 'dataset/test.h5' # path to h5 dataset (required)
    SPLIT = 'dataset/splits.json' # path to split file (required)
    SPLIT_ID = 0 # split index (default: 0)
    METRIC = 'summe' # evaluation metric ['tvsum', 'summe'])

    # Model options
    #INPUT_DIM = 1024#2048 # input dimension (default: 1024)
    INPUT_DIM = 2048 # input dimension (default: 1024)
    HIDDEN_DIM = 256 # hidden unit dimension of DSN (default: 256)
    NUM_LAYERS = 1 # number of RNN layers (default: 1)
    RNN_CELL = 'lstm' # RNN cell type (default: lstm)

    # Optimization options
    LR = 1e-05 # learning rate (default: 1e-05)
    WEIGHT_DECAY = 1e-05 # weight decay rate (default: 1e-05)
    MAX_EPOCH = 60 # maximum epoch for training (default: 60)
    STEP_SIZE = 30 # how many steps to decay learning rate (default: 30)
    GAMMA = 0.1 # learning rate decay (default: 0.1)
    NUM_EPISODE = 5 # number of episodes (default: 5)
    BETA = 0.01 # weight for summary length penalty term (default: 0.01)

    # Misc
    SEED = 1 # random seed (default: 1)
    GPU = '0' # which gpu devices to use (default: 0)
    USE_CPU = False # use cpu device
    EVALUATE = False # whether to do evaluation only
    TEST = False # whether to do evaluation only
    RESUME = '' # path to resume file
    VERBOSE = True # whether to show detailed test results
    SAVE_DIR = 'model/' # path to save output (default: log/)
    SAVE_RESULTS = True # whether to save output results