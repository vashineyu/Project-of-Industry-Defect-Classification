"""
lots of other callbacks function
"""
from PIL import Image
import cv2
import numpy as np
import scipy.stats 
import csv
from scipy.stats import percentileofscore
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score, f1_score

from keras.callbacks import Callback
from keras.models import load_model, save_model

### customized functions
def img_center_crop(img, target_size):
    # return center cropprd image (not resizing)
    # img should be a PIL image object
    # target size should be a tuple, eg (224, 224)
    width, height = img.size
    if width <= target_size[0] and height <= target_size[1]:
        return img
    left = (width - target_size[0])/2
    right = (width + target_size[0])/2
    top = (height - target_size[1])/2
    bottom = (height + target_size[1])/2
    
    new_img = img.crop((left, top, right, bottom))
    return new_img

def im_preCrop_90(img):
    im = Image.open(img)
    im_out = im.crop((5, 5, 95, 95))
    im_out = im_out.resize((100, 100))
    return im_out

def self_im_preproc(img, do_it = False):
    if do_it:
        im_self_mean = img.mean(axis = (0,1))
        im_out = img - im_self_mean
    else:
        im_out = img
        
    return im_out

class LossHistory(Callback):
    def on_train_begin(self,logs={}):
        self.loss=[]
        self.val_loss=[]
    def on_epoch_end(self,epoch,logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))

### compute AUC for the validation set at the end of each epoch
class LogAUC(Callback):
    def __init__(self):
        #self.aucs = []
        return

    def on_train_begin(self, logs={}):    
        # dir(self.model)
        return

    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        #    if "auc" not in self.params['metrics']:
        #       self.params['metrics'].append("auc")
        if "val_auc" not in self.params['metrics']:
            self.params['metrics'].append("val_auc")
            
    def on_epoch_end(self, epoch, logs = {}):
        logs = logs or {}
        #y_pred = self.model.predict(self.validation_data[0])
        y_pred = self.model.predict(self.validation_data[0])
        
        #self.aucs.append(auc)
        logs["val_auc"] = roc_auc_score(self.validation_data[1], y_pred)

class f1sc(Callback):
    def __init__(self, fname = 'tmp.csv'):
        self.fname = fname
        return

    def on_train_begin(self, logs={}):    
        return

    def on_epoch_begin(self, epoch, logs = {}):
        logs = logs or {}
        if "val_f1sc" not in self.params['metrics']:
            # f1sc
            self.params['metrics'].append("val_f1sc")
        if "val_fp" not in self.params['metrics']:
            # false-positive (false alarm)
            self.params['metrics'].append("val_fp")
        if "val_fn" not in self.params['metrics']:
            # false-negative (miss)
            self.params['metrics'].append("val_fn")
        if "val_tp" not in self.params['metrics']:
            # true-positive (hit)
            self.params['metrics'].append("val_tp")
        if "val_tn" not in self.params['metrics']:
            # trun-negative (correct reject)
            self.params['metrics'].append("val_tn")
            
    def on_epoch_end(self, epoch, logs = {}):
        logs = logs or {}
        thres = 0.5 # prepare it, for further we can change our judgement threshold
        y_true = self.validation_data[1].argmax(axis = 1)
        y_pred = self.model.predict(self.validation_data[0])
        
        #import csv
        """
        with open(self.fname, 'a+') as f:
            writer = csv.writer(f)
            writer.writerows([y_pred[:,1]])
        """
        ###
        y_pred = (y_pred[:, 1] > thres) * 1
        
        con_martrix = confusion_matrix(y_true= y_true, y_pred= y_pred)
        
        ### 0 1 (pred)
        # 0
        # 1
        # (true)
        
        tp = con_martrix[1][1]
        tn = con_martrix[0][0]
        fp = con_martrix[0][1]
        fn = con_martrix[1][0]
        
        logs["val_f1sc"] = f1_score(y_true = y_true, y_pred = y_pred)
        logs["val_tp"] = tp
        logs["val_tn"] = tn
        logs["val_fp"] = fp
        logs["val_fn"] = fn
        
class WeightHistory(Callback):
    def __init__(self, layer_name):
        self.layer_name = layer_name
        self.rate = []

    def on_batch_begin(self, batch, logs={}):
        self.weight = self.model.get_layer(self.layer_name).get_weights()[0]
        self.param_scale = np.linalg.norm(self.weight.ravel())

    def on_batch_end(self, batch, logs={}):
        self.update = self.model.get_layer(self.layer_name).get_weights()[0] - self.weight
        self.update_scale = np.linalg.norm(self.update.ravel())
        self.rate.append(self.update_scale/self.param_scale)

### make parallel
from keras.layers import merge
from keras.layers.core import Lambda
from keras.models import Model
import tensorflow as tf

def make_parallel(model, gpu_count):
    def get_slice(data, idx, parts):
        shape = tf.shape(data)
        size = tf.concat([ shape[:1] // parts, shape[1:] ],axis=0)
        stride = tf.concat([ shape[:1] // parts, shape[1:]*0 ],axis=0)
        start = stride * idx
        return tf.slice(data, start, size)

    outputs_all = []
    for i in range(len(model.outputs)):
        outputs_all.append([])

    #Place a copy of the model on each GPU, each getting a slice of the batch
    for i in range(gpu_count):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('tower_%d' % i) as scope:

                inputs = []
                #Slice each input into a piece for processing on this GPU
                for x in model.inputs:
                    input_shape = tuple(x.get_shape().as_list())[1:]
                    slice_n = Lambda(get_slice, output_shape=input_shape, arguments={'idx':i,'parts':gpu_count})(x)
                    inputs.append(slice_n)                

                outputs = model(inputs)
                
                if not isinstance(outputs, list):
                    outputs = [outputs]
                
                #Save all the outputs for merging back together later
                for l in range(len(outputs)):
                    outputs_all[l].append(outputs[l])

    # merge outputs on CPU
    with tf.device('/cpu:0'):
        merged = []
        for outputs in outputs_all:
            merged.append(merge(outputs, mode='concat', concat_axis=0))
            
        return Model(input=model.inputs, output=merged)

