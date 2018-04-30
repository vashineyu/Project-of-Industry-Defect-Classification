# usage
"""
dic = {'non_copper_meta_all': 'path_to_Non_Copper_all.csv',
       'copper_meta_all': 'path_to_Copper_all.csv',
       'non_copper_testing': 'path_to_Non_Copper_testing.csv',
       'copper_testing': 'path_to_Copper_testing.csv',
       'non_copper_img_folder' : 'path_to_non_copper_img_folder, #/home/gtx980/Desktop/CCP/Data/DataSet/All/'
       'copper_img_folder': 'path_to_copper_img_folder, #/home/gtx980/Desktop/CCP/Data/DataSet/Copper/',
       'valid_ratio' : 0.1}
tmp_class = init_data(dic)
train_nonC, val_nonC, test_nonC, train_C, val_C, test_C = tmp_class.get_train_val_test_df()
"""
import glob
import os
import numpy as np
import pandas as pd
import random
import pandas as pd
from sklearn.model_selection import train_test_split


# init data
class init_data():
    def __init__(self, input_dict):
        # f: file
        # d: dir
        # path to the csv (filename)
        self.df_non_copper_all = pd.read_csv(input_dict['non_copper_meta_all'])
        self.df_copper_all = pd.read_csv(input_dict['copper_meta_all'])
        self.df_non_copper_testing = pd.read_csv(input_dict['non_copper_testing'])
        self.df_copper_testing = pd.read_csv(input_dict['copper_testing'])
        # path of image folder
        self.d_non_copper = input_dict['non_copper_img_folder']
        self.d_copper = input_dict['copper_img_folder']
        
        # run init func
        self._make_list()
        #self.get_train_val_test_df()
    
    def get_train_val_test_df(self):
        # return train / validation / test data frame
        non_copper_trainset, non_copper_validset = train_test_split(self.df_non_copper_train,
                                                                   test_size = 1 - (0.8/0.9) )
        non_copper_train = non_copper_trainset.reset_index(drop = True)
        non_copper_validset = non_copper_validset.reset_index(drop = True)
        
        copper_trainset, copper_validset = train_test_split(self.df_copper_train,
                                                           test_size = 1 - (0.8/0.9) )
        copper_train = copper_trainset.reset_index(drop = True)
        copper_validset = copper_validset.reset_index(drop = True)
        
        return (non_copper_trainset, non_copper_validset, self.df_non_copper_testing, 
                copper_trainset, copper_validset, self.df_copper_testing)
        
    def _make_list(self):
        # generate availale list and keep it
        self.df_non_copper_all['im_path'] = self.d_non_copper + self.df_non_copper_all['pid']
        # non-copper-set
        # remove from testing
        tmp = list(set(self.df_non_copper_all.pid) - set(self.df_non_copper_testing.pid) )
        self.df_non_copper_train = self.df_non_copper_all[self.df_non_copper_all.pid.isin(tmp)]
        # remove from those without pic (make sure we have the image)
        tmp = glob.glob(self.d_non_copper + "/*.png")
        tmp = [os.path.basename(i) for i in tmp]
        self.df_non_copper_train = self.df_non_copper_train[self.df_non_copper_all.pid.isin(tmp)]
        self.df_non_copper_train = self.df_non_copper_train.reset_index(drop = True)
        self.df_non_copper_testing = self.df_non_copper_testing[self.df_non_copper_testing.pid.isin(tmp)]
        self.df_non_copper_testing = self.df_non_copper_testing.reset_index(drop = True)
        self.df_non_copper_testing['im_path'] = self.d_non_copper + self.df_non_copper_testing['pid']
        
        # copper-set
        self.df_copper_all['im_path'] = self.d_copper + self.df_copper_all['pid']
        tmp = list(set(self.df_copper_all.pid) - set(self.df_copper_testing.pid) )
        self.df_copper_train = self.df_copper_all[self.df_copper_all.pid.isin(tmp)]
        tmp = glob.glob(self.d_copper + "/*.png")
        tmp = [os.path.basename(i) for i in tmp]
        self.df_copper_train = self.df_copper_train[self.df_copper_all.pid.isin(tmp)]
        self.df_copper_testing = self.df_copper_testing[self.df_copper_testing.pid.isin(tmp)]
        self.df_copper_train = self.df_copper_train.reset_index(drop = True)
        self.df_copper_testing = self.df_copper_testing.reset_index(drop = True)
        self.df_copper_testing['im_path'] = self.d_copper + self.df_copper_testing['pid']
        

class init_data_from_directory():
    def __init__(self, params):
        self.d_class0_train = params['dir_train']['d_class0']
        self.d_class1_train = params['dir_train']['d_class1']
        self.d_class0_test = params['dir_test']['d_class0']
        self.d_class1_test = params['dir_test']['d_class1']
        
        if '' in params['dir_valid'].values():
            try:
                print('use inner split k-folds')
                self.flag_inner_split = True
                self.valid_ratio = params['valid_ratio']
            except KeyError:
                print("Using validation from train dir need to set up a valid_ratio")
        else:
            self.flag_inner_split = False
            self.d_class0_valid = params['dir_valid']['d_class0']
            self.d_class1_valid = params['dir_valid']['d_class1']
        
        # run init function
        self._build_list()
    
    def get_train_val_test_df(self):
        return(self.df_class0_train, self.df_class0_valid, self.df_class0_test,
               self.df_class1_train, self.df_class1_valid, self.df_class1_test)
    
    def _build_list(self):
        # glob path and make it a list (id - path)
        
        if self.flag_inner_split:
            # validaion set should drawn from training set folder
            tmp = glob.glob(self.d_class0_train + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            i_train = random.sample(list(np.arange(len(tmp))), int(len(tmp) * (1-self.valid_ratio) ))
            
            iid = [tid[x] for x in i_train]
            iip = [tmp[x] for x in i_train]
            
            self.df_class0_train = pd.DataFrame({'pid' : iid,
                                                'im_path' : iip, 
                                                'is_cooper_defect' : 'N'})
            iid = list(set(tid) - set(iid) )
            iip = list(set(tmp) - set(iip) )
            
            self.df_class0_valid = pd.DataFrame({'pid' : iid,
                                                 'im_path' : iip,
                                                 'is_cooper_defect' : 'N'})
            
            tmp = glob.glob(self.d_class1_train + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            i_train = random.sample(list(np.arange(len(tmp))), int(len(tmp) * (1-self.valid_ratio) ))
            
            iid = [tid[x] for x in i_train]
            iip = [tmp[x] for x in i_train]
            
            self.df_class1_train = pd.DataFrame({'pid' : iid,
                                                 'im_path' : iip,
                                                 'is_cooper_defect' : 'Y'})
            iid = list(set(tid) - set(iid) )
            iip = list(set(tmp) - set(iip) )
            
            self.df_class1_valid = pd.DataFrame({'pid' : iid,
                                                 'im_path' : iip,
                                                 'is_cooper_defect' : 'Y'})
            #
        else:
            tmp = glob.glob(self.d_class0_train + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            self.df_class0_train = pd.DataFrame({'pid' : tid,
                                                 'im_path' : tmp,
                                                 'is_cooper_defect' : 'N'})
            
            tmp = glob.glob(self.d_class1_train + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            self.df_class1_train = pd.DataFrame({'pid' : tid,
                                                'im_path' : tmp,
                                                'is_cooper_defect' : 'Y'})
            
            tmp = glob.glob(self.d_class0_valid + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            self.df_class0_valid = pd.DataFrame({'pid' : tid,
                                                'im_path' : tmp,
                                                'is_cooper_defect' : 'N'})
            
            tmp = glob.glob(self.d_class1_valid + '/*.png')
            tid = [os.path.basename(i) for i in tmp]
            self.df_class1_valid = pd.DataFrame({'pid' : tid,
                                                'im_path' : tmp,
                                                'is_cooper_defect' : 'Y'})
        
        # for testing set
        tmp = glob.glob(self.d_class0_test + '/*.png')
        tid = [os.path.basename(i) for i in tmp]
        self.df_class0_test = pd.DataFrame({'pid' : tid,
                                            'im_path' : tmp,
                                            'is_cooper_defect' : 'N'})
        
        tmp = glob.glob(self.d_class1_test + '/*.png')
        tid = [os.path.basename(i) for i in tmp]
        self.df_class1_test = pd.DataFrame({'pid' : tid,
                                            'im_path' : tmp,
                                            'is_cooper_defect' : 'Y'})
        
            
        