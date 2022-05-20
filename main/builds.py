# add folder to system path
# import sys
# sys.path.append("../../../")

import os 
import random as rn
import tensorflow as tf
import logging

# to shut up tensorflow misc warnings
tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

import cv2
import numpy as np
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
from tensorflow.keras.regularizers import l1_l2

import ep_generator as epgen

# import scripts.gpu_setup
tf.compat.v1.disable_eager_execution()
# tf.compat.v1.enable_eager_execution()

##########################################################################
##########################################################################

def setSeed(seed=42):
    # set reproducibility seed
    # note that is impossible to reproduce exactly equal results with GPUs
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    rn.seed(seed)    
    
    return None

def ResblockA(input_layer, num):
    activation = layers.ReLU
    BNtraining = False
    x = layers.Conv2D(num, (3, 3), padding="same")(input_layer)
    x = layers.BatchNormalization()(x, training=BNtraining)
    x = activation()(x)
    x = layers.Conv2D(num, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x, training=BNtraining)
    x = layers.Add()([x, input_layer])
    x = activation()(x)    
    return x

def ResblockB(input_layer, num):
    activation = layers.ReLU
    BNtraining = False
    x1 = layers.Conv2D(num, (3, 3), padding="same")(input_layer)
    x1 = layers.BatchNormalization()(x1, training=BNtraining)
    x1 = activation()(x1)
    x1 = layers.Conv2D(num, (3, 3), padding="same")(x1)
    x1 = layers.BatchNormalization()(x1, training=BNtraining)
    
    x2 = layers.Conv2D(num, (1, 1), padding="same")(input_layer)
    x2 = layers.BatchNormalization()(x2, training=BNtraining)
    
    
    
    x = layers.Add()([x1, x2, input_layer])
    x = activation()(x)
    
    return x

#############################################################################
l1 = 1e-7
l2 = 1e-6
def conv2D_block(layer, num, K_size=(3, 3), padding='same', BN=True, activation=layers.ReLU):
    x = layers.Conv2D(num, K_size, padding=padding, kernel_regularizer=None)(layer)
    if BN:
        x = layers.BatchNormalization()(x, training=False)
    if activation:
        x = activation()(x)

    return x

def DenseBlock(input_layer, num):   
    conv3_0 = conv2D_block(input_layer, num, (3, 3))
    conv1_0 = conv2D_block(input_layer, num, (1, 1))    
    x1 = layers.Add()([conv3_0, conv1_0])
    x1 = conv2D_block(x1, num)
    
    x2 = conv2D_block(conv3_0, num, (1, 1))
    x2 = layers.Add()([x1, x2])
            
    return x2
    
    
def build_IEM(input_shape):
    activation = layers.ReLU
    input_layer = layers.Input(input_shape, name="InputLayer")
    x = conv2D_block(input_layer, 16, (3, 3))
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = DenseBlock(x, 16)    
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = DenseBlock(x, 32)    
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = conv2D_block(x, 64, (3, 3))
    model = Model(input_layer, x, name='IEM')
    return model
    
def build_RM(input_shape):
    activation = layers.ReLU
    input_layer = layers.Input(input_shape, name="InputLayer")
    x = conv2D_block(input_layer, 16, (3, 3))
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = DenseBlock(x, 32)
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = DenseBlock(x, 32)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(8)(x)
    x = layers.BatchNormalization()(x)
    x = activation()(x)
    x = layers.Dense(1, activation="sigmoid")(x)    
    model = Model(input_layer, x, name='RM')
    return model


def build_CTM(input_shape):    
    activation = layers.ReLU
    input_layer = layers.Input(input_shape)
    conv3_0 = conv2D_block(input_layer, 32, (3, 3))
    conv1_0 = conv2D_block(input_layer, 32, (1, 1))
    x = layers.Add()([conv3_0, conv1_0])    
    x = conv2D_block(x, 32)
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = layers.Conv2D(32, (3, 3), padding="same", activation="sigmoid")(x)        
    return Model(input_layer, x, name="CTM")
    

def build_RESHAPER(input_shape):    
    activation = layers.ReLU
    input_layer = layers.Input(input_shape)
    conv3_0 = conv2D_block(input_layer, 32, (3, 3))
    conv1_0 = conv2D_block(input_layer, 32, (1, 1))
    x = layers.Add()([conv3_0, conv1_0])
    x = conv2D_block(x, 32, (3, 3))
    x = layers.MaxPooling2D(strides=(2, 2))(x)
    x = conv2D_block(x, 32)
    return Model(input_layer, x, name="RESHAPER")
    


def build_RN(anom_shots,
             good_shots,
             query_shots,
             from_base=None):
    
    if not from_base:
        EM = build_IEM((320, 320, 1))    
        CTM = build_CTM((*EM.output_shape[1:-1], EM.output_shape[-1]*2))
        RESHAPER = build_RESHAPER(EM.output_shape[1:])
        RM = build_RM((*CTM.output_shape[1:-1], CTM.output_shape[-1]*2))
    else:
        baseModel = load_model(from_base)
        EM = [layer for layer in baseModel.layers if layer.name == 'IEM'][0]
        CTM = [layer for layer in baseModel.layers if layer.name == 'CTM'][0]
        RESHAPER = [layer for layer in baseModel.layers if layer.name == 'RESHAPER'][0]
        RM = [layer for layer in baseModel.layers if layer.name == 'RM'][0]
    
    inputs = []
    embeddings = []
    for i in range(anom_shots + good_shots + query_shots):
        inputs.append(layers.Input(EM.input.shape[1:], name='input_' + str(i)))
        embeddings.append(EM(inputs[-1]))

    # split embedding classes
    embeddings_anom = embeddings[: anom_shots]
    embeddings_good = embeddings[anom_shots: good_shots + anom_shots]
    embeddings_query = embeddings[-query_shots:]

    # make the fusion layers (few-shot) in the support set (anomalies and good)
    fusion_anom  = layers.Average(name='Fusion_Anom')(embeddings[: anom_shots])
    fusion_good  = layers.Average(name='Fusion_Good')(embeddings[anom_shots: good_shots + anom_shots])
    
    # CTM implementation
    concentrated = layers.Concatenate()([fusion_anom, fusion_good])
    mask = CTM(concentrated)
    
    # Reshaper
    reshape_fusion_anom = RESHAPER(fusion_anom)
    reshape_fusion_good = RESHAPER(fusion_good)
    reshape_queries = []
    for query in embeddings_query:
        reshape_queries.append(RESHAPER(query))
    
    # Improved features
    improved_anom = layers.Multiply()([reshape_fusion_anom, mask])
    improved_good = layers.Multiply()([reshape_fusion_good, mask])
    improved_queries = []
    for query in reshape_queries:
        improved_queries.append(layers.Multiply()([query, mask]))

    # concatenate the two fusion layers with each query embedding
    concat_anom_query = []
    concat_good_query = []
    for i, query in enumerate(improved_queries):
        concat_anom_query.append(layers.Concatenate(name='Concat_Anom_' + str(i))([improved_anom, query]))
        concat_good_query.append(layers.Concatenate(name='Concat_Good_' + str(i))([improved_good, query]))

    # pass every relational feature through the relation module g(x)
    relations_anom = []
    relations_good = []
    for relation_anom_query, relation_good_query in zip(concat_anom_query, concat_good_query):
        relations_anom.append(RM(relation_anom_query))
        relations_good.append(RM(relation_good_query))

    # a lambda layer for put together the two relation scores (<anom vs query> and <good vs query>)
    def concat_relations(items):
        relations_anom = tf.concat(items[0], axis=-1)
        relations_good = tf.concat(items[1], axis=-1)

        return tf.stack([relations_anom, relations_good], axis=-1)
    relation_output = layers.Lambda(concat_relations)([relations_anom, relations_good])


    return Model(inputs, relation_output, name='RelationNet')    

def build_trainer_v1(model, eta, query_shots):
    def loss_fn(y_true, y_pred):
        l2 = lambda a: tf.sqrt(tf.reduce_sum(tf.pow(a, 2)))
        exp = tf.exp
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        pp = l2(y_true[:, :, 0] - y_pred[:, :, 0])
        pn = l2(y_true[:, :, 0] - y_pred[:, :, 1])
        np = l2(y_true[:, :, 1] - y_pred[:, :, 0])
        nn = l2(y_true[:, :, 1] - y_pred[:, :, 1])

        p_contrast = exp(pp) / (exp(pp) + exp(pn))
        n_contrast = exp(nn) / (exp(nn) + exp(np))
        sample_loss = tf.pow(l2([p_contrast, n_contrast - 1]), 2)
        mean_loss = tf.reduce_mean(sample_loss)
        
        return mean_loss
    
    
    eta = tf.constant(eta)
    inputs = model.inputs
    labels = layers.Input((query_shots, 2))
    
    pred = model(inputs, training=True)    
    loss = loss_fn(labels, pred)
    eta_value = tf.get_static_value(eta)
    updates = Adam(eta_value).get_updates(loss, model.trainable_weights)
    trainer = K.function([inputs, labels, eta], loss, updates)    

    print("...RN model + trainer built.")    
    return trainer


def build_trainer_v2(model, eta, query_shots):
    exp = tf.exp
    def loss_fn(y_true, y_pred):
        BCE = binary_crossentropy(y_true, y_pred, from_logits=True)
        BCE_mean = tf.reduce_mean(BCE)
        
        return BCE_mean
    
    eta = tf.constant(eta)
    inputs = model.inputs
    labels = layers.Input((query_shots, 2))
    pred = model(inputs, training=True)        

    loss = loss_fn(labels, pred)
    
    eta_value = tf.get_static_value(eta)
    updates = Adam(eta_value).get_updates(loss, model.trainable_weights)
    trainer = K.function([inputs, labels, eta], loss, updates)    

    print("...RN model + trainer built.")    
    return trainer, loss_fn
