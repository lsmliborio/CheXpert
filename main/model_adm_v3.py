#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os 
import random as rn
import tensorflow as tf
import logging

# to shut up tensorflow misc warnings
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

import cv2
import numpy as np
import pandas as pd
import os
import glob
import time
import pydot
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm, trange

from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras import layers                                    
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model

import ep_generator as epgen
import builds as B
# import scripts.gpu_setup
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

class Model_adm():
    class Config():        
        class RN():
            input_shape = 1
            eta = 1
            positive_shots = 1
            negative_shots = 1
            query_shots = 1               
            eta_reduce_factor = 1
            eta_reduce_schedule = 0
            batch_size = 1
            epochs = 1

    class Builds():        
        build_RN = staticmethod(B.build_RN)
        build_trainer = staticmethod(B.build_trainer_v2)
        
    def fit(self, train_data, val_data, model, target, trainer, loss_fn):        
        B.setSeed()
        
        positive_shots = self.Config.RN.positive_shots
        negative_shots = self.Config.RN.negative_shots
        query_shots = self.Config.RN.query_shots
        eta = self.Config.RN.eta
        batch_size = self.Config.RN.batch_size
        epochs = self.Config.RN.epochs        
        eta_reduce_factor = self.Config.RN.eta_reduce_factor
        eta_reduce_schedule = self.Config.RN.eta_reduce_schedule
        
        positive, negative = train_data[target]
        
        log_train = []
        log_val = []
        try:            
            for epoch in trange(epochs + 1):                         
                X, y = epgen.fetch_RN(positive, negative,
                                      positive_shots,
                                      negative_shots,
                                      query_shots,
                                      batch_size)
                loss = (trainer([X, y, eta]))

                if epoch % (epochs // 100) == 0:
                    print(f"epoch: {epoch:5d} │ Loss: {loss:+.3f}")
                    
                if eta_reduce_schedule > 0 and epoch % eta_reduce_schedule == 0 and epoch > 0:
                    eta *= eta_reduce_factor
                    print(f"\n\nREDUCING ETA TO... {eta:.6f}", end='\n\n')        
                    
                
                # validation log                
                if epoch % (epochs // 10) == 0:
                    log_train_mid = []
                    log_val_mid = []
                    
                    train_positive, train_negative = train_data[target]
                    val_positive, val_negative = val_data[target]
                    for i in range(20):                         
                        X_train, y_train = epgen.fetch_RN(train_positive, train_negative,
                                                          positive_shots,
                                                          negative_shots,
                                                          query_shots,
                                                          batch_size)
                        X_val, y_val = epgen.fetch_RN(val_positive, val_negative,
                                                      positive_shots,
                                                      negative_shots,
                                                      query_shots,
                                                      batch_size)
                        pred_train = model.predict(X_train)
                        pred_val = model.predict(X_val)
                        
                        train_loss = loss_fn(y_train, pred_train)
                        val_loss = loss_fn(y_val, pred_val)      
                        
                        log_train_mid.append(train_loss)
                        log_val_mid.append(val_loss)

                    log_train.append((K.eval(tf.reduce_mean(log_train_mid))))    
                    log_val.append(K.eval(tf.reduce_mean(log_val_mid)))

            return log_train, log_val
        except KeyboardInterrupt:
            print(f"Training interrupted at epoch {epoch}.")
            return log_train, log_val         

    
    def evaluate(self, data, target, model, iterations):    
        B.setSeed()
        
        batch_size = self.Config.RN.batch_size
        positive_shots = self.Config.RN.positive_shots
        negative_shots = self.Config.RN.negative_shots
        query_shots = self.Config.RN.query_shots
        positive, negative = data[target]
        
        print("Evaluating model...\n")
        log = []
        for i in trange(iterations):
            inner_log = []            
            for i in range(10):                                
                X, y = epgen.fetch_RN(positive, negative,
                                       positive_shots,
                                       negative_shots,
                                       query_shots,
                                       batch_size)            

#                 pred = K.eval(model(X, training=True))
                pred = model.predict(X)
                test = np.argmax(y, axis=2) == np.argmax(pred, axis=2)    
                test = (test * 1)

                inner_log.append(test)
            log.append(np.array(inner_log).mean())

        log = np.array(log)
        mean_acc = log.mean()
        std_acc = log.std()
        
        print(f"Accuracy: {round(mean_acc*100, 2)} +- {round(std_acc*1.96*100, 2)}%")            

        return mean_acc, std_acc
    
    
    def evaluateTransfer(self, data, target, model, iterations, batch_size, 
                         positive_shots, negative_shots, query_shots):    
        B.setSeed()
        positive, negative = data[target]
        print("Evaluating model...\n")
        log = []
        for i in trange(iterations):
            inner_log = []            
            for i in range(10):                                
                X, y = epgen.fetch_RN(positive, negative,
                                       positive_shots,
                                       negative_shots,
                                       query_shots,
                                       batch_size)            

                pred = model.predict(X)
                test = np.argmax(y, axis=2) == np.argmax(pred, axis=2)    
                test = (test * 1)

                inner_log.append(test)
            log.append(np.array(inner_log).mean())

        log = np.array(log)
        mean_acc = log.mean()
        std_acc = log.std()
        
        print(f"Accuracy: {round(mean_acc*100, 2)} +- {round(std_acc*1.96*100, 2)}%")            

        return mean_acc, std_acc


# In[2]:


MAIN_PATH = "CheXpert-v1.0-small"
TRAIN_PATH = os.path.join(MAIN_PATH, 'train')
VALID_PATH = os.path.join(MAIN_PATH, 'valid')
TRAIN_CSV_PATH = os.path.join(MAIN_PATH, 'train_v3.csv')
VALID_CSV_PATH = os.path.join(MAIN_PATH, 'valid_v3.csv')

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_valid = pd.read_csv(VALID_CSV_PATH)

full_data_train = epgen.get_full_data(TRAIN_CSV_PATH)
full_data_valid = epgen.get_full_data(VALID_CSV_PATH)

print("Train shapes")
for key in full_data_train.keys():
    p, n = full_data_train[key]
    print(key, p.shape, n.shape)
    
print("\nValidation shapes")
del_keys = []
for key in full_data_valid.keys():
    p, n = full_data_valid[key]
    
    if p.shape[0] <= 20 or n.shape[0] <= 20:
        del_keys.append(key)
    else:
        print(key, p.shape, n.shape)
        
for key in del_keys:
    full_data_valid.pop(key)


# In[14]:


# start model administrator object
model_adm = Model_adm()
builds = model_adm.Builds
config_RN = model_adm.Config.RN

# configure setup
config_RN.input_shape = (320, 320, 1)
config_RN.eta = 3e-4
config_RN.positive_shots = 5
config_RN.negative_shots = 5
config_RN.query_shots = 3
config_RN.batch_size = 64
config_RN.epochs = 15000
config_RN.eta_reduce_factor = 0.5
config_RN.eta_reduce_schedule = 10000

# generate models and trainers
# RN = builds.build_RN(config_RN.positive_shots, 
#                      config_RN.negative_shots,
#                      config_RN.query_shots)
# trainer, loss_fn = builds.build_trainer(RN,
#                                config_RN.eta,
#                                config_RN.query_shots)


# In[4]:


RN.summary()


# In[7]:


full_data_valid.keys()


# In[25]:


target = 'Fracture'
path = "models/" + target + "_v3.h5py"
# config_RN.eta = 5e-5
# config_RN.batch_size = 56
config_RN.epochs = 8000
# config_RN.eta_reduce_factor = 0.75
config_RN.eta_reduce_schedule = 0
train_log3, val_log3 = model_adm.fit(full_data_train, full_data_train, RN, target, trainer, loss_fn)


# In[28]:


model_adm.evaluate(full_data_train, target, RN,  2)


# In[48]:


RN.save(path)


# In[49]:


positive_shots = 8
negative_shots = 8
query_shots = 2
transferRN = builds.build_RN(positive_shots, negative_shots, query_shots, path)
model_adm.evaluateTransfer(full_data_train, target, transferRN,  2, 16, positive_shots, negative_shots, query_shots)


# RN = load_model("models/" + target + "_v1.h5py")
# model_adm.evaluate(full_data_train, target, RN,  3)
