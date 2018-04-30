import csv
import pandas as pd
import numpy as np
import scipy as sp
import argparse
import glob
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--model", type = str, help = "path to model.h5", default = None)
parser.add_argument('-l',"--im_folder", nargs = '*', help = "path to png files", required = False)
parser.add_argument("--out_csv", type = str, help = "Path & filename for the output csv", default = "./test.csv")
parser.add_argument("--dataset_mean", type = str, help = "file contain rgb of dataset", default = None)
parser.add_argument("--csv_file", type = str, help = "If using csv identifier to make prediction", default = None)
parser.add_argument("--out_layer", type = str, help = "If output from previous layer, give the output layer name", default = None)
parser.add_argument("--gpu_device_num", type= str, help = "Decide which CUDA_VISIBLE_DEVICES will be used", default = None)
args = parser.parse_args()

# global_average_pooling2d_2

from PIL import Image
import skimage.io as skio
import os

if args.gpu_device_num is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device_num

import keras
from keras.models import Model, load_model
from keras.applications.resnet50 import preprocess_input
from callbacks_miscs import *

### function for -training set mean
def self_mean_preproc(im, d_mean):
    im[:,:,:,0] -= d_mean[0]
    im[:,:,:,1] -= d_mean[1]
    im[:,:,:,2] -= d_mean[2]
    im = im[:, :, :, ::-1] # reorder to BGR
    return im

### function for unifying image load
def op_img(x):
    X_ = np.array([ np.array(img_center_crop(Image.open(i), target_size=(100, 100)), dtype='float32') for i in x] )
    X_ = np.array([scipy.misc.imresize(i, size=(200, 200)) for i in X_]) # add this line to resize images
    X_ = X_.reshape((len(X_), 200, 200, 3))
    return X_

def img_crop_patch(x, crop_size, target_size):
    w, h = x.size
    cw, ch = crop_size # e.g. 60, 60
    
    lx1, ly1 = 0 + cw, 0 + ch # patch one: left top
    lx2, ly2 = 0 + cw, h - ch # patch two: left bottom
    lx3, ly3 = w - cw, 0 + ch # patch three: right top
    lx4, ly4 = w - cw, h - ch # patch four: right bottom
    lx5_1, lx5_2, ly5_1, ly5_2 = (w - cw)//2, w - (w - cw)//2, (h - ch)//2, h - (h - ch)//2
    
    im_1 = x.crop((0, 0, lx1, ly1)).resize(target_size)
    im_2 = x.crop((0, ly2, lx2, h)).resize(target_size)
    im_3 = x.crop((lx3, 0, w, ly3)).resize(target_size)
    im_4 = x.crop((lx4, ly4, w, h)).resize(target_size)
    im_5 = x.crop((lx5_1, ly5_1, lx5_2, ly5_2)).resize(target_size)
    
    return x, im_1, im_2, im_3, im_4, im_5

### read images
if args.csv_file is None:
    print('predict png directly from folder')
    img_path = []
    for in_dir in args.im_folder:
        img_path.extend(glob.glob(in_dir + "/*.png"))
    #img_path = glob.glob(args.im_folder + "/*.png")
else:
    print('use csv file to grab png files')
    im_list = pd.read_csv(args.csv_file, header = True)

if args.dataset_mean is None:
    print("Use imagenet mean")
else:
    print("Use dataset mean")
    ## load the txt file and make it into RGB
    with open(args.dataset_mean, 'r') as f:
        _tmp = f.readlines()
        dataset_mean = [np.float32(i.split('\n')[0]) for i in _tmp]
        print('rgb values of dataset: ' + str(dataset_mean) )

print(args.im_folder)
print('Numbers of images to predict: ' + str(len(img_path) ))

# make image size judgement (is 100px or 200 px?)
# separate prediction if there are too many
model = load_model(args.model)
if args.out_layer != None:
    print("output from previous layer: " + args.out_layer)
    model = Model(inputs = [model.input], outputs = [model.get_layer(args.out_layer).output])
else:
    print('output: full')

# stop if image is not 100 x 100
if Image.open(img_path[0]).size != (100, 100):
    print("Check your image size first, the image size should be 100 x 100")
    sys.exit()

# go throgh images by each image
y_pred = list()
counter = 0
for this_im in img_path:
    imgs = img_crop_patch(Image.open(this_im), (75, 75), (100, 100))
    imgs = [np.array(i) for i in imgs]
    imgs = np.array([scipy.misc.imresize(i, size=(200, 200)) for i in imgs]) # add this line to resize images
    imgs = imgs.astype('float32')
    
    if args.dataset_mean is None:
        imgs = preprocess_input(imgs)
    else:
        # use loaded dataset mean to do the proc
        imgs = self_mean_preproc(imgs, dataset_mean)
    keeper = model.predict(imgs)
    keeper = keeper.mean(axis = 0)
    y_pred.append(keeper)
    if counter % 100 == 0:
        print(counter)
    counter += 1
print(np.array(y_pred).shape)
result = pd.DataFrame({'png_name': img_path, 'y_pred': np.squeeze(np.array(y_pred))[:,1] })
result.to_csv(args.out_csv)

"""
# should do resize (check size part)
if Image.open(img_path[0]).size != (200, 200):
    
    # total numbers to predict (in the folder path)
    total_cnts = len(img_path)
    total_run = int(np.ceil(total_cnts / 10000.0)) # ceil (how many parts can we split)
    #print(total_run)
    if total_cnts <= 10000:
        imgs = op_img(img_path)
        #imgs = np.array([np.array(Image.open(i).resize((200, 200), Image.BILINEAR )) for i in img_path])
        imgs = imgs.astype('float32')

        if args.dataset_mean is None:
            # use imagenet mean to do the preprocessing
            imgs = preprocess_input(imgs)
        else:
            # use loaded dataset mean to do the proc
            imgs = self_mean_preproc(imgs, dataset_mean)
            
        y_pred = model.predict(imgs)
        tag = False
    else:
        #
        y_pred = list()
        for ind in np.arange(total_run):
            ind = int(ind)
            print('runing index: ' + str(ind) )
            imgs = op_img(img_path[ind * 10000 : (ind + 1) * 10000 ])
            #imgs = np.array([ np.array(Image.open(i).resize((200, 200), Image.BILINEAR )) for i in img_path[ind * 10000 : (ind + 1) * 10000 ] ])
            imgs = imgs.astype('float32')
            if args.dataset_mean is None:
                imgs = preprocess_input(imgs)
            else:
                # use loaded dataset mean to do the proc
                imgs = self_mean_preproc(imgs, dataset_mean)
            
            keeper = model.predict(imgs)
            keeper = list(keeper[:, 0:])
            y_pred.extend(keeper)
            tag = True
else:
    # don't do resize
    if total_cnts <= 10000:
        #imgs = np.array([np.array(Image.open(i) ) for i in img_path])
        imgs = op_img(img_path)
        imgs = imgs.astype('float32')
        if args.dataset_mean is None:
            imgs = preprocess_input(imgs)
        else:
            # use loaded dataset mean to do the proc
            imgs = self_mean_preproc(imgs, dataset_mean)
        
        y_pred = model.predict(imgs)
        tag = False
    else:
        y_pred = list()
        for ind in np.arange(total_run):
            print(ind)
            ind = int(ind)
            print('runing index: ' + str(ind) )
            imgs = op_img(img_path[ind * 10000 : (ind + 1) * 10000 ])
            #imgs = np.array([ np.array(Image.open(i) ) for i in img_path[ind * 10000 : (ind + 1) * 10000 ] ])
            imgs = imgs.astype('float32')
            if args.dataset_mean is None:
                imgs = preprocess_input(imgs)
            else:
                # use loaded dataset mean to do the proc
                imgs = self_mean_preproc(imgs, dataset_mean)

            keeper = model.predict(imgs)
            print(keeper)
            keeper = list(keeper[:, 0:])
            y_pred.extend(keeper)
            tag = True
            
print('---------')
print(len(y_pred))
#print(y_pred.shape)

if tag: # means it is larger than 10000 or not
    if args.out_layer is None:
        result = pd.DataFrame({'png_name': img_path, 'y_pred': np.squeeze(y_pred) })
    else:
        result = pd.DataFrame({'png_name': img_path})
        _tmp = pd.DataFrame(y_pred)
        result = pd.concat([result, _tmp], axis = 1)
else:
    if args.out_layer is None:
        #print(np.squeeze(y_pred))
        #print(np.squeeze(y_pred).shape)
        result = pd.DataFrame({'png_name': img_path, 'y_pred': np.squeeze(y_pred)[:,1]})
        # y_pred[:,1]
    else:
        ### TO DO ###
        # FIX THE BUG. Still a bug here, don't use the intermediate output layer here#
        ############
        result = pd.DataFrame({'png_name': img_path})
        _tmp = pd.DataFrame(y_pred)
        result = pd.concat([result, _tmp], axis = 1)

result.to_csv(args.out_csv)

try:
    print('good')
except AttributeError:
    # prevent newer version sess error
    print('pass')
"""