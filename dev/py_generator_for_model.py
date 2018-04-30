
import random
from PIL import Image
import glob
import numpy as np
import scipy as sp
import scipy.stats 
import pandas as pd
import os
import csv
import time

from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score
from multiprocessing import Pool, Pipe

from callbacks_miscs import *
from keras.utils import np_utils
from keras.applications.resnet50 import preprocess_input
from keras.models import load_model, save_model

class call_generators():
    def __init__(self, params_dict, dta_gen = None):
        self.tags = params_dict['tags']
        self.crop_w, self.crop_h = params_dict['crop_w'], params_dict['crop_h']
        self.img_w, self.img_h = params_dict['img_w'], params_dict['img_h']
        self.channels = params_dict['img_channels']
        self.dta_gen = dta_gen
        self.use_self_improc = params_dict['use_self_improc']
        self.dataset_mean = params_dict['dataset_mean']

    def train_generator(self, bz, 
                        df_class0, df_class1,
                        class_0_ratio = 1, class_1_ratio = 1,
                        action = 'shuffle'):
        # bz: batch size
        # df_class0: non-copper
        # df_class1: copper
        # action: shuffle or random
        def use_shuffle(list, start, end):
            array = []
            for k in range(start, end):
                array.append(list[k])
            return array

        while(1):
            per_get = bz / len(self.tags)
            # decide how to get data
            if action == 'shuffle':
                # use shuffle to make mini-batch
                
                for start in range(0, len(df_class0)*10, int(bz/2)):
                    idx_0_start = int(start % len(df_class0))
                    idx_0_end = int((start + bz/2) % len(df_class0))
                    if idx_0_start > idx_0_end:
                        X_train_0 = use_shuffle(np.array(df_class0.im_path), idx_0_start - len(df_class0), idx_0_end)
                        Y_train_0 = use_shuffle(np.array(df_class0.is_cooper_defect), idx_0_start - len(df_class0), idx_0_end)
                    else:
                        X_train_0 = df_class0.im_path[idx_0_start:idx_0_end]
                        Y_train_0 = df_class0.is_cooper_defect[idx_0_start:idx_0_end]
                    idx_1_start = int(start % len(df_class1))
                    idx_1_end = int((start + bz/2) % len(df_class1))
                    if idx_1_start > idx_1_end:
                        X_train_1 = use_shuffle(np.array(df_class1.im_path), idx_1_start - len(df_class1), idx_1_end)
                        Y_train_1 = use_shuffle(np.array(df_class1.is_cooper_defect), idx_1_start - len(df_class1), idx_1_end)
                    else:
                        X_train_1 = df_class1.im_path[idx_1_start:idx_1_end]
                        Y_train_1 = df_class1.is_cooper_defect[idx_1_start:idx_1_end]
                    X_train = np.append(X_train_1, X_train_0)
                    Y_train = np.append(Y_train_1, Y_train_0)
                    
                    # start reading 
                    """
                    X_train = np.array([ np.array(img_center_crop(Image.open(i), 
                                                      target_size=(self.crop_w, 
                                                                   self.crop_h)), 
                                      dtype='float32') for i in X_train] )
                    X_train = np.array([scipy.misc.imresize(i, 
                                                    size=(self.img_w, 
                                                          self.img_h)) for i in X_train]) # add this line to resize images
                    X_train = X_train.reshape((len(X_train), 
                                       self.img_w, self.img_h, self.channels))
                    """
                    X_train = op_img(X_train, (self.crop_w, self.crop_h), (self.img_w, self.img_h, self.channels))
            
                    # y to dummies
                    le = preprocessing.LabelEncoder()
                    le.fit(self.tags) # N as 0, Y as 1
                    Y_train = le.transform(Y_train)
                    Y_train = np_utils.to_categorical(Y_train, len(self.tags))
            
                    if self.dta_gen is None:
                       pass
                    else:
                        datagen = self.dta_gen
                        datagen.fit(X_train)
                        for X_train, Y_train in datagen.flow(X_train, 
                                                     Y_train, 
                                                     batch_size = len(X_train),
                                                     #save_to_dir= 'imgs_tmp', save_prefix='aug', save_format='jpeg'
                                                    ):
                    # go one loop
                            break
                    
                    X_train = X_train.astype('float32')
                    if self.use_self_improc:
                        if self.use_self_improc == 'dataset':
                            #print('use dataset mean')
                            X_train[:,:,:,0] -= self.dataset_mean[0]
                            X_train[:,:,:,1] -= self.dataset_mean[1]
                            X_train[:,:,:,2] -= self.dataset_mean[2]
                        else:
                            #print('use image self mean')
                            X_train = np.array([self_im_preproc(i, True) for i in X_train])
                        # move the dimension to BGR by yourself!
                        X_train = X_train[:, :, :, ::-1] # tensorflow way
                    else:
                        #print('use data mean')
                        X_train = preprocess_input(X_train)
            
                    yield X_train, Y_train
                    ####
            else:
                X_train = []
                Y_train = []
                # train different per_get here --> can be 1 : 2
                i_get_0 = df_class0.sample(n = int(per_get * class_0_ratio))
                i_get_1 = df_class1.sample(n = int(per_get * class_1_ratio))
                i_get = pd.concat((i_get_0, i_get_1))

                X_train = list(i_get.im_path)
                Y_train = list(i_get.is_cooper_defect)
            
                # start reading 
                """
                X_train = np.array([ np.array(img_center_crop(Image.open(i), 
                                                      target_size=(self.crop_w, 
                                                                   self.crop_h)), 
                                      dtype='float32') for i in X_train] )
                X_train = np.array([scipy.misc.imresize(i, 
                                                    size=(self.img_w, 
                                                          self.img_h)) for i in X_train]) # add this line to resize images
                X_train = X_train.reshape((len(X_train), 
                                       self.img_w, self.img_h, self.channels))
            """
                X_train = op_img(X_train, (self.crop_w, self.crop_h), (self.img_w, self.img_h, self.channels))
                # y to dummies
                le = preprocessing.LabelEncoder()
                le.fit(self.tags) # N as 0, Y as 1
                Y_train = le.transform(Y_train)
                Y_train = np_utils.to_categorical(Y_train, len(self.tags))
            
                if self.dta_gen is None:
                    pass
                else:
                    datagen = self.dta_gen
                    datagen.fit(X_train)
                    for X_train, Y_train in datagen.flow(X_train, 
                                                     Y_train, 
                                                     batch_size = len(X_train),
                                                     #save_to_dir= 'imgs_tmp', save_prefix='aug', save_format='jpeg'
                                                    ):
                    # go one loop
                        break
                    
                X_train = X_train.astype('float32')
                if self.use_self_improc:
                    if self.use_self_improc == 'dataset':
                    #print('use dataset mean')
                        X_train[:,:,:,0] -= self.dataset_mean[0]
                        X_train[:,:,:,1] -= self.dataset_mean[1]
                        X_train[:,:,:,2] -= self.dataset_mean[2]
                    else:
                        #print('use image self mean')
                        X_train = np.array([self_im_preproc(i, True) for i in X_train])
                # move the dimension to BGR by yourself!
                    X_train = X_train[:, :, :, ::-1] # tensorflow way
                else:
                    #print('use data mean')
                    X_train = preprocess_input(X_train)
            
                yield X_train, Y_train
            
    def get_validation_data(self,
                            df_class0, df_class1,
                            class_0_ratio = 1, 
                            use_im_gen = None,
                            n_gen_loop = 5):
        # should always take all class_1 (copper) images
        X_ = []
        Y_ = []
        per_get = len(df_class1)

        i_get_0 = df_class0.sample(n = int(per_get * class_0_ratio))
        i_get_1 = df_class1
        i_get = pd.concat((i_get_0, i_get_1))
        
        X_ = list(i_get.im_path)
        Y_ = list(i_get.is_cooper_defect)
        print(len(X_))
        X_ = np.array([ np.array(img_center_crop(Image.open(i), 
                                                      target_size=(self.crop_w, 
                                                                   self.crop_h)), 
                                      dtype='float32') for i in X_] )
        X_ = np.array([scipy.misc.imresize(i, 
                                                size=(self.img_w, 
                                                      self.img_h)) for i in X_]) # add this line to resize images
        X_ = X_.reshape((len(X_), 
                         self.img_w, self.img_h, self.channels))
            
        # y to dummies
        #return Y_val
        le = preprocessing.LabelEncoder()
        le.fit(self.tags) # N as 0, Y as 1
        Y_ = le.transform(Y_)
        Y_ = np_utils.to_categorical(Y_, len(self.tags))
        
        if use_im_gen is None:
            pass
        else:
            datagen = use_im_gen
            datagen.fit(X_)
            i_start = 1
            i_end = n_gen_loop # run 5 time and return original image + 5 times aug (x6)
            if i_end > 0:
                for X_gen, Y_gen in datagen.flow(X_, Y_, batch_size = len(X_), shuffle = False):
                    X_ = np.concatenate((X_, X_gen))
                    Y_ = np.concatenate((Y_, Y_gen))
                    if i_start == i_end:
                        break
                    i_start += 1
        print(X_.shape)
        
        X_ = X_.astype('float32')
        if self.use_self_improc:
            if self.use_self_improc == 'dataset':
                print('use dataset mean')
                X_[:,:,:,0] -= self.dataset_mean[0]
                X_[:,:,:,1] -= self.dataset_mean[1]
                X_[:,:,:,2] -= self.dataset_mean[2]
            else:
                print('use image self mean')
                X_ = np.array([self_im_preproc(i, True) for i in X_])
            X_ = X_[:, :, :, ::-1] # tensorflow way
        else:
            print('use imagenet mean')
            X_ = preprocess_input(X_)

        return X_, Y_
    
        
        return
    
    def model_predict_testing(self, model_name, df_class0, df_class1, testing_batch_size = 5000):
        import time
        df_all = pd.concat((df_class0, df_class1))
        df_all = df_all.reset_index(drop = True)
        total_counts = len(df_all) / np.float(testing_batch_size)
        print(total_counts)
        #result = pd.DataFrame({'y_pred': None,
        #                      'y_true': None,
        #                      'png_name': None})
        y_pred_keeper = []
        name_keeper = []
        truth_keeper = []
        model = load_model(model_name)

        for ind in np.arange(total_counts):
            ind = int(ind)
            print("runung index: " + str(ind))

            X_ = list(df_all[ind * testing_batch_size : (ind + 1) * testing_batch_size].im_path)
            Y_ = list(df_all[ind * testing_batch_size : (ind + 1) * testing_batch_size].is_cooper_defect)
            
            tmp_name_keeper = X_
            

            X_ = np.array([ np.array(img_center_crop(Image.open(i), 
                                                  target_size=(self.crop_w, self.crop_h)), 
                                  dtype='float32') for i in X_] )
            X_= np.array([scipy.misc.imresize(i, 
                                              size=(self.img_w,
                                                    self.img_h)) 
                          for i in X_])
            X_= X_.reshape((len(X_), 
                            self.img_w, self.img_h, self.channels))

            # y to dummies
            le = preprocessing.LabelEncoder()
            le.fit(self.tags) # N as 0, Y as 1
            Y_ = le.transform(Y_)
            Y_ = np_utils.to_categorical(Y_, len(self.tags))

            X_ = X_.astype('float32')
            if self.use_self_improc:
                if self.use_self_improc == 'dataset':
                    #print('use dataset mean')
                    X_[:,:,:,0] -= self.dataset_mean[0]
                    X_[:,:,:,1] -= self.dataset_mean[1]
                    X_[:,:,:,2] -= self.dataset_mean[2]
                else:
                    X_ = np.array([self_im_preproc(i, True) for i in X_])
                X_ = X_[:, :, :, ::-1]
            else:
                X_ = preprocess_input(X_)
            
            tmp_truth_keeper = Y_[:,1]
            
            t1 = time.time()
            y_pred = model.predict(X_)
            t2 = time.time()
            print(t2 - t1)
            
            y_pred = list(y_pred[:,1]) # probability of predicting 1 
            
            y_pred_keeper.append(y_pred)
            name_keeper.append(tmp_name_keeper)
            truth_keeper.append(tmp_truth_keeper)
            
        aout, bout, cout = [], [], []
        for i in y_pred_keeper:
            aout.extend(i)
        for i in truth_keeper:
            bout.extend(i)
        for i in name_keeper:
            cout.extend(i)
        """
        result = pd.DataFrame({'y_pred' : y_pred_keeper,
                               'y_true' : truth_keeper,
                               'png_name': name_keeper})
        """
        result = pd.DataFrame({'y_pred' : aout,
                               'y_true' : bout,
                               'png_name': cout})
        return result       
    
    def __testing_generator(self, df_class0, df_class1, testing_batch_size = 5000):
        ####
        # NEVER USE IT 
        ####
        df_all = pd.concat((df_class0, df_class1))
        df_all = df_all.reset_index(drop = True)
        total_counts = len(df_all) / np.float(testing_batch_size)
        while (1):
            
            for ind in np.arange(total_counts):
                ind = int(ind)
                
                X_ = list(df_all[ind * testing_batch_size : (ind + 1) * testing_batch_size].im_path)
                Y_ = list(df_all[ind * testing_batch_size : (ind + 1) * testing_batch_size].is_cooper_defect)
                
                X_ = np.array([ np.array(img_center_crop(Image.open(i), 
                                                      target_size=(self.crop_w, 
                                                                   self.crop_h)), 
                                      dtype='float32') for i in X_] )
                X_= np.array([scipy.misc.imresize(i, 
                                                  size=(self.img_w,
                                                        self.img_h)) 
                              for i in X_train])
                X_= X_.reshape((len(X_), 
                                self.img_w, self.img_h, self.channels))

                # y to dummies
                le = preprocessing.LabelEncoder()
                le.fit(self.tags) # N as 0, Y as 1
                Y_ = le.transform(Y_)
                Y_ = np_utils.to_categorical(Y_, len(self.tags))
                
                X_ = X_.astype('float32')
                X_ = preprocess_input(X_)
                
                yield X_, y_

### functions to calculate training set mean
def image_mean(img):
        # img: image list
        # return: img mean [r,g,b]
        X_ = np.array(Image.open(img))
        im_mean = X_.mean(axis = (0,1))
        return im_mean

def ave_rgb(x):
    # x: img_path
    # return cumulative [r,g,b] 
    tmp_rgb = np.zeros([1,3])
    for j in x:
        tmp_rgb += image_mean(j)
    return tmp_rgb

def split_into_multi_list(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last+avg)])
        last += avg
    return out

def get_training_set_mean(df_class0, df_class1, n_core):

    df_all = pd.concat((df_class0, df_class1))
    df_all = df_all.reset_index(drop = True)

    X_ = list(df_all.im_path)

    #rgb_mean = np.zeros([1,3]) # set first value

    #counter = 0
    #for ix in X_:
    #    rgb_mean += image_mean(ix)
    #    counter += 1
    #    if counter % 10000 == 0:
    #        print('Current progress: ' + str(counter) )

    p = Pool(n_core)
    print('start computing')
    t1 = time.time()
    tmp_out = p.map(ave_rgb, split_into_multi_list(X_, 5))
    rgb_mean = np.array(tmp_out).sum(axis = (0,1))
    
    # append rgb, then divided by total numbers
    rgb_mean = rgb_mean / len(df_all)
    # return dataset (training set) mean of rgb channels
    print('compute end')
    t2 = time.time()
    print(t2 - t1)

    return rgb_mean

def op_img(x, crop_size, output_size):
        # crop size: w, h
        # output size, w, h, c
        crop_w, crop_h = crop_size
        out_w, out_h, out_c = output_size
        X_ = np.array([ np.array(img_center_crop(Image.open(i), 
                                                      target_size=(crop_w, crop_h)), dtype='float32') for i in x] )
        X_ = np.array([scipy.misc.imresize(i, size=(out_w, out_h)) for i in X_]) # add this line to resize images
        X_ = X_.reshape((len(X_), out_w, out_h, out_c))
        return X_