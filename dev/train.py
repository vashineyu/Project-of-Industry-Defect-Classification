# python3 main script
"""
General usage: 
On linux: CUDA_VISIBLE_DEVICES=4 python3 train.py --i_fold 1
"""

# basic libraries
from __future__ import print_function
import argparse
import numpy as np
import pandas as pd
import os
import time

#import tensorflow as tf
#from keras.backend.tensorflow_backend import set_session
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.8
#set_session(tf.Session(config = config))

### add argument parsing ###

parser = argparse.ArgumentParser()
parser.add_argument("--i_fold", type = str, help = "fold number to append on model name", default = '1')
parser.add_argument("--n_gen_loop", type = int, help = "Validation set augment ratio (loop)", default = 3)

args = parser.parse_args()
i_fold = args.i_fold

# NN libraries
import keras
from keras.models import Sequential, Model
from keras.models import load_model, save_model
from keras.optimizers import SGD, Adam, Adagrad # gradient
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
K.set_image_dim_ordering('tf')

# customized functions (should under the same path)
from callbacks_miscs import *
from py_init_data import *
from py_generator_for_model import *

# there should be a config.py under the same path
from config import *
print('Config import complete')

if nn_activation == 'relu':
    ### If use default resnet
    from keras.applications.resnet50 import ResNet50
else:
    ### If want to modify the resnet structure
    from resnet_with_drp import *

from keras.models import Sequential, Model, load_model, save_model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Conv2D, BatchNormalization
from keras.layers import AveragePooling2D, ZeroPadding2D, GlobalAveragePooling2D

def resnet_model_build(resnet_model, use_stage, freeze_stage, acti,
                       use_merge = False, 
                       n_meta = 0,
                       fc_drop_rate = 0.2):
    #if use merge should always check n_meta
    
    fc_drop_rate = float(fc_drop_rate)
    for layer in resnet_model.layers:
        layer.trainable = True
    #resnet_model.summary()

    # design for using different activation function that will change the layer name
    if acti == 'relu':
        to_get = 'activation_'
    else:
        to_get = acti + "_"

    if use_stage == 1:
        get_layer = "max_pooling2d_2"
    elif use_stage == 2:
        #get_layer = "activation_10"
        get_layer = to_get + '10'
    elif use_stage == 3:
        #get_layer = "activation_22"
        get_layer = to_get + '22'
    elif use_stage == 4:
        #get_layer = "activation_40"
        get_layer = to_get + '40'
    else:
        get_layer = "global_avgerage_pooling2d_1"

    if freeze_stage == 1:
        free_layer_num = 5
    elif freeze_stage == 2:
        free_layer_num = 37
    elif freeze_stage == 3:
        free_layer_num = 79
    elif freeze_stage == 4:
        free_layer_num = 141
    else:
        free_layer_num = 176

    if freeze_stage == 0:
        print('all parameter tunable')
    else:
        for layer in resnet_model.layers[:free_layer_num]:
            layer.trainable = False
        
    if use_stage != 5:    
        x = resnet_model.get_layer(get_layer).output
        #x = AveragePooling2D((13, 13), name='avg_pool')(x)
        #x = Flatten()(x)
        x = GlobalAveragePooling2D()(x)
    else:
        x = resnet_model.get_layer(get_layer).output
    
    if use_merge:
        meta_info = Input(shape = (n_meta, )) # n_meta: numbers of features from meta
        x = keras.layers.concatenate([x, meta_info])
    else:
        pass
    
    """
    x = Dense(64, name = 'dense1')(x)
    x = BatchNormalization(axis = -1, name = 'dense1_bn')(x)
    x = Activation('relu', name = 'dense1_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd1_drop')(x)
    
    x = Dense(32, name = 'dense2')(x)
    x = BatchNormalization(axis = -1, name = 'dense2_bn')(x)
    x = Activation('relu', name = 'dense2_activation')(x)
    x = Dropout(fc_drop_rate, name = 'd2_drop')(x)
    """
    
    out = Dense(2, activation="softmax", name = "output")(x)
        
    model_final = Model(inputs = [resnet_model.input], outputs = [out])
    return model_final


"""
Train all data, cut by date
"""
print('script start')
# Define names for saved model name
opt = model_output_prefix  + '_k' + str(i_fold)
model_file_name = dir_out_model + "/model_" + opt + ".h5"

# Initialize the data
data_cla = init_data_from_directory(data_params)
train_nonC, val_nonC, test_nonC, train_C, val_C, test_C = data_cla.get_train_val_test_df()
print('Non_copper training/validation/testing ' + str(len(train_nonC)) + "/" + str(len(val_nonC)) + "/" + str(len(test_nonC)))
print('copper training/validation/testing ' + str(len(train_C)) + "/" + str(len(val_C)) + "/" + str(len(test_C)))

# Check table independcy here (should be empty!, if not empty, it means data contamination)
if len(set(train_C.pid).intersection(test_C.pid) ) != 0:
    print('die')
    raise 'YOU MUST ERROR HERE!'
if len(set(train_nonC.pid).intersection(test_nonC.pid) ) != 0:
    print('die')
    raise 'YOU MUST ERROR HERE!'
        
# Get training set mean of rgb
if generator_params_dict['use_self_improc'] == 'dataset':
    print('use dataset mean')
    avg_dataset = get_training_set_mean(df_class0= train_C, df_class1= train_nonC, n_core=8)
    generator_params_dict['dataset_mean'] = avg_dataset / np.float32(dataset_mean_ratio)
    # write the self_mean information to a txt file
    csv_file_name = dir_out_model + "/rgbConfig_" + opt + ".txt"
    np.savetxt(csv_file_name, avg_dataset)
else:
    generator_params_dict['dataset_mean'] = None
    print('do not use dataset mean')

# Initiaalize the generator
gen_data = call_generators(generator_params_dict, dta_gen= datagen)
x_val, y_val = gen_data.get_validation_data(df_class0= val_nonC, df_class1= val_C,
                                                    class_0_ratio = 1,  use_im_gen = datagen_val, n_gen_loop = args.n_gen_loop)
K.clear_session()
if nn_activation == 'relu':
    print('use defaut activation function')
    resnet_model = ResNet50(include_top=False, weights = "imagenet", input_shape = (200, 200, 3), pooling ='avg')
else:
    print('use activation function: ' + nn_activation)
    resnet_model = ResNet50(include_top=False, weights = "imagenet", input_shape = (200, 200, 3), pooling ='avg', acti = nn_activation)

model = resnet_model_build(resnet_model, freeze_stage= fs, use_stage= us, acti = nn_activation)
if n_gpu_use > 1:
    print('use ' + str(n_gpu_use) + ' to merging')
    model = make_parallel(model, n_gpu_use)

model.summary()

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6)
myoptimizer = Adam(lr= lr)
model.compile(loss='binary_crossentropy', optimizer=myoptimizer, metrics=['acc'])

earlystop = EarlyStopping(monitor= 'val_loss', 
                                  min_delta= 0.0001, 
                                  patience= nb_epoch / 10, 
                                  verbose=0, mode='auto')

checkpoint = ModelCheckpoint(model_file_name,
                                     monitor='val_loss',
                                     verbose=1,
                                     save_best_only=True,
                                     mode='auto')
loss_history = LossHistory()
if use_merge:
    pass
else:
    history_model = model.fit_generator(gen_data.train_generator(df_class0=train_nonC, 
                                                                     df_class1=train_C,
                                                                     class_0_ratio = 1,
                                                                     class_1_ratio = 1,
                                                                     bz = batch_size, action = mini_batch_method),
                                            steps_per_epoch = n_batch,
                                            epochs= nb_epoch,
                                            validation_data=(x_val, y_val),
                                            callbacks = [
                                                         reduce_lr,
                                                         loss_history, 
                                                         checkpoint, 
                                                         earlystop,
                                                         LogAUC(), 
                                                         f1sc()
                                                         
                                            ])

    # save training process
train_loss = history_model.history.get("loss")
train_acc = history_model.history.get("acc")
val_loss = history_model.history.get("val_loss")
val_acc = history_model.history.get("val_acc")
val_auc = history_model.history.get("val_auc")
val_f1 = history_model.history.get('val_f1sc')
val_tp = np.array(history_model.history.get('val_tp')).astype('float32')
val_tn = np.array(history_model.history.get('val_tn')).astype('float32')
val_fp = np.array(history_model.history.get('val_fp')).astype('float32')
val_fn = np.array(history_model.history.get('val_fn')).astype('float32')
    
pd_tmp = pd.DataFrame({'train_loss': train_loss,
                               'valid_loss': val_loss,
                               'train_acc': train_acc,
                               'valid_acc': val_acc,
                               'valid_f1': val_f1,
                               'valid_auc': val_auc,
                               'valid_TP': val_tp,
                               'valid_TN': val_tn,
                               'valid_FP': val_fp,
                               'valid_FN': val_fn})
pd_tmp.to_csv(opt + '_training_process.csv')
# make prediction
pred_out = gen_data.model_predict_testing(model_name = model_file_name, 
                                                  df_class0 = test_nonC, 
                                                  df_class1 = test_C, 
                                                  testing_batch_size= 10000)
pred_out.to_csv(dir_out_csv + '/testing_' + opt + '.csv', index = False)
print('a round end')
