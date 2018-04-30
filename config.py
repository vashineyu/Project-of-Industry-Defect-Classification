# config -- put parameters here
from keras.preprocessing.image import ImageDataGenerator

### fold number and naming
#i_fold = 1 # fold number on naming
model_output_prefix = 'resnetMean_DateCut10_DeployTestVer' # Remember to modify the parameter below, this line is only about file naming
### controling parameters
n_gpu_use = 1 # this should compatible with CUDA_VISIBLE_DEVICES number

### model related parameters
fs = 0 # layer to freeze 
us = 4 # layers to dump
lr = 0.00017 # learning rate at begin
drp = 0 # dropout ratio in the Resnet Conv layers
batch_size = 64 * n_gpu_use
nb_epoch = 150 # numbers of epoch for the training process
n_batch = 400 # numbers of updates per epoch
use_merge = False # Ignore it, design to merge meta-data
mini_batch_method = "shuffle" # shuffle or random
nn_activation = 'relu' # activation type in the resnet (default should be relu, option: relu / leakyrelu / elu)
dataset_mean_ratio = 1 # it matter if use dataset mean

### data information
dir_out_csv = '/home/seanyu/project/CCP/res_csv/' # result csv output location
dir_out_model = '/home/seanyu/project/CCP/model/' # model output location

### data initialize parameters
# dir_train: training set location
# dir_valid: validation set location (if leave blank, automatically get val set from training set by valid_ratio)
# dir_test: testing set location
data_params = {
            'dir_train': {'d_class0': '/data/put_data/seanyu/ccp/clean_date_cut/thres10/non_copper_train/',
                                'd_class1': '/data/put_data/seanyu/ccp/clean_date_cut/thres10/copper_train/'
                 },
            # leave white space " " as value if there is no validation dir
            'dir_valid': {'d_class0': '',
                                'd_class1': ''
                 },
            'dir_test': {'d_class0': '/data/put_data/seanyu/ccp/clean_date_cut/thres10/non_copper_test/',
                               'd_class1': '/data/put_data/seanyu/ccp/clean_date_cut/thres10/copper_test/'
                 },
            'valid_ratio' : 0.1
        }

### model input information
# tags: is copper defect or not (Y: copper, N: non-copper, watchout: the ordering trap ... alphabet ordering)
# crop_w/h: crop size from input image
# img_w/h: image size for the model (resizing)
# img_channels: RGB = 3
# use_self_improc: True (-selfmean) / False (-imagenet mean) / dataset (-dataset mean)
generator_params_dict = {'tags' : ['N', 'Y'],
                                 'crop_w': 100,
                                 'crop_h': 100,
                                 'img_w': 200,
                                 'img_h': 200,
                                 'img_channels': 3,
                                 'use_self_improc' : False # True / False / 'dataset'
                                }

### parameters for train generator
# parameters reference: https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0,
        zoom_range=[0.2, 2.3],
        horizontal_flip=True, vertical_flip = True,
        fill_mode='wrap')

### parameters for validation augmentation
datagen_val = None
"""
datagen_val = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0,
    zoom_range=[0.5, 1.5],
    horizontal_flip=True, vertical_flip = True,
    fill_mode='wrap')
"""