
# coding: utf-8

# In[ ]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential, Model
from keras.layers import Activation, Dense, Flatten, Input, Layer, MaxPooling2D, Conv2D, Reshape, UpSampling2D
from keras.losses import binary_crossentropy, mse, mean_absolute_error
from keras import backend as K
from keras.optimizers import Adam
from keras.constraints import unit_norm
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

import tensorflow as tf
import numpy as np
import scipy.stats as st
import pandas as pd
import seaborn as sns

from keras import regularizers
import sys
import argparse
import os
import glob 

D1 = 64
D2 = 32
L2 = 0.01


# In[ ]:


def build_encoder_T1(N1):
    input_data = Input(shape=(N1,))
    output_data = Dense(D1, activation='tanh',kernel_regularizer=regularizers.l2(L2))(input_data)
    output_data = Dense(D2, activation='tanh',kernel_regularizer=regularizers.l2(L2))(output_data)
    
    return Model(input_data, output_data)

def build_encoder_DTI(N2):
    input_data = Input(shape=(N2,))
    output_data = Dense(D1, activation='tanh',kernel_regularizer=regularizers.l2(L2))(input_data)
    output_data = Dense(D2, activation='tanh',kernel_regularizer=regularizers.l2(L2))(output_data)
    
    return Model(input_data, output_data)


# In[ ]:


def build_decoder_T1(N1):
    input_data = Input(shape=(D2,))
    output_data = Dense(D1, activation='tanh',kernel_regularizer=regularizers.l2(L2))(input_data)
    output_data = Dense(N1, activation='tanh',kernel_regularizer=regularizers.l2(L2))(output_data)
     
    return Model(input_data, output_data)

def build_decoder_DTI(N2):
    input_data = Input(shape=(D2,))
    output_data = Dense(D1, activation='tanh',kernel_regularizer=regularizers.l2(L2))(input_data)
    output_data = Dense(N2, activation='tanh',kernel_regularizer=regularizers.l2(L2))(output_data)
     
    return Model(input_data, output_data)


# In[ ]:


def build_ae_T1(encoder, decoder,N1):
    input_image_ae_1 = Input(shape=(N1,))
    input_image_ae_2 = Input(shape=(N1,))
    input_dummy = Input(shape=(1,))

    feature_enc_1 = encoder(input_image_ae_1)
    feature_enc_2 = encoder(input_image_ae_2)

    d = Dense(D2)(input_dummy)

    reconstruct_image_1 = decoder(feature_enc_1)
    reconstruct_image_2 = decoder(feature_enc_2)

    return Model([input_image_ae_1,input_image_ae_2,input_dummy], 
                 [reconstruct_image_1,reconstruct_image_2,feature_enc_1,feature_enc_2,d])

def build_ae_DTI(encoder, decoder,N2):
    input_image_ae_1 = Input(shape=(N2,))
    input_image_ae_2 = Input(shape=(N2,))
    input_dummy = Input(shape=(1,))

    feature_enc_1 = encoder(input_image_ae_1)
    feature_enc_2 = encoder(input_image_ae_2)

    d = Dense(D2)(input_dummy)

    reconstruct_image_1 = decoder(feature_enc_1)
    reconstruct_image_2 = decoder(feature_enc_2)

    return Model([input_image_ae_1,input_image_ae_2,input_dummy], 
                 [reconstruct_image_1,reconstruct_image_2,feature_enc_1,feature_enc_2,d])

def build_LSSL(ae,N1):
    input_1 = Input(shape=(N1,))
    input_2 = Input(shape=(N1,))
    input_dummy = Input(shape=(1,))

    [recon_1,recon_2,feat_1,feat_2,d] = ae([input_1,input_2,input_dummy])

    lssl_1 = Model([input_1,input_2,input_dummy],[recon_1,recon_2,d])

    ## loss
    recon_loss_1 = mse(input_1,recon_1)
    recon_loss_2 = mse(input_2,recon_2)

    long_vec = feat_2 - feat_1
    long_vec_dis = K.sqrt(K.sum(K.square(long_vec),axis=-1))
    d_len = K.sqrt(K.sum(K.square(d),axis=-1))
    pd = K.sum(long_vec * d, axis=-1)
    cos_loss = K.tf.divide(pd,(long_vec_dis*d_len))

    lssl_loss =  K.mean(recon_loss_1+recon_loss_2-2*cos_loss + 2)
    lssl_1.add_loss(lssl_loss)
    
    return lssl_1

def build_LCA(ae1,ae2,N1,N2):
    input1_1 = Input(shape=(N1,))
    input1_2 = Input(shape=(N1,))
    input2_1 = Input(shape=(N2,))
    input2_2 = Input(shape=(N2,))
    input_dummy = Input(shape=(1,))

    [recon1_1,recon1_2,feat1_1,feat1_2,d1] = ae1([input1_1,input1_2,input_dummy])
    [recon2_1,recon2_2,feat2_1,feat2_2,d2] = ae2([input2_1,input2_2,input_dummy])

    joint_ae = Model([input1_1,input1_2,input2_1,input2_2,input_dummy], 
                     [recon1_1,recon1_2,recon2_1,recon2_2,d1,d2], 
                     name='LCA')

    recon_loss1_1 = mse(input1_1,recon1_1)
    recon_loss1_2 = mse(input1_2,recon1_2)

    recon_loss2_1 = mse(input2_1,recon2_1)
    recon_loss2_2 = mse(input2_2,recon2_2)

    long_vec1 = feat1_2 - feat1_1
    d1_len = K.sqrt(K.sum(K.square(d1),axis=-1))
    pd1 = K.tf.divide(K.sum(long_vec1 * d1, axis=-1),d1_len)

    long_vec2 = feat2_2 - feat2_1
    d2_len = K.sqrt(K.sum(K.square(d2),axis=-1))
    pd2 = K.tf.divide(K.sum(long_vec2 * d2, axis=-1),d2_len)

    #alignment_loss = K.square(pd1-pd2)
    mpd1 = K.mean(pd1)
    mpd2 = K.mean(pd2)
    pd1m = pd1-mpd1
    pd2m = pd2-mpd2
    r_num = K.sum(tf.multiply(pd1m,pd2m))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(pd1m)), K.sum(K.square(pd2m)))) + 1e-10
    r = K.tf.divide(r_num, r_den)
    #r = K.maximum(K.minimum(r, 1.0), -1.0)
    alignment_loss = 1-K.square(r)

    ae_loss =  K.mean(recon_loss1_1+recon_loss1_2+
                      recon_loss2_1+recon_loss2_2)+4*alignment_loss

    joint_ae.add_loss(ae_loss)

    return joint_ae

def build_CCA(encoder1_cca,encoder2_cca,decoder1_cca,decoder2_cca,N1,N2):
    input_1_cca = Input(shape=(N1,))
    input_2_cca = Input(shape=(N2,))
    input_dummy = Input(shape=(1,))

    feat_1_cca = encoder1_cca(input_1_cca)
    feat_2_cca = encoder2_cca(input_2_cca)

    d1_cca = Dense(D2)(input_dummy)
    d2_cca = Dense(D2)(input_dummy)

    recon_1_cca = decoder1_cca(feat_1_cca)
    recon_2_cca = decoder2_cca(feat_2_cca)

    cca = Model([input_1_cca,input_2_cca,input_dummy], 
            [recon_1_cca,recon_2_cca,d1_cca,d2_cca], 
            name='CCA')
    ##
    recon_loss1_cca = mse(input_1_cca,recon_1_cca)
    recon_loss2_cca = mse(input_2_cca,recon_2_cca)

    d1_len = K.sqrt(K.sum(K.square(d1_cca),axis=-1))
    pd1 = K.tf.divide(K.sum(feat_1_cca * d1_cca, axis=-1),d1_len)

    d2_len = K.sqrt(K.sum(K.square(d2_cca),axis=-1))
    pd2 = K.tf.divide(K.sum(feat_2_cca * d2_cca, axis=-1),d2_len)

    #alignment_loss = K.square(pd1-pd2)
    mpd1 = K.mean(pd1)
    mpd2 = K.mean(pd2)
    pd1m = pd1-mpd1
    pd2m = pd2-mpd2
    r_num = K.sum(tf.multiply(pd1m,pd2m))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(pd1m)), K.sum(K.square(pd2m)))) + 1e-6
    r = K.tf.divide(r_num, r_den)
    #r = K.maximum(K.minimum(r, 1.0), -1.0)
    alignment_loss = 1-K.square(r)

    cca_loss =  K.mean(recon_loss1_cca+recon_loss2_cca)+2*alignment_loss

    cca.add_loss(cca_loss)
    
    return cca
