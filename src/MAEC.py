#!/usr/bin/env python
# coding: utf-8

# In[15]:
import os

import numpy as np
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, GRU, Layer, Attention, LeakyReLU, Reshape, Softmax, Dropout, Conv1D, Concatenate, GlobalAveragePooling1D, MaxPooling1D, LeakyReLU, Flatten, multiply, Activation, Add, BatchNormalization, Bidirectional, TimeDistributed
import tensorflow.keras.backend as K
import pickle as pkl

from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from utilz import *

class MAEC():
    def __init__(self):
        self.latent_dim = 2048
        self.mil_spec_shape = (70, 50,45)
        
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer='adam',
                                   metrics=['acc'])
        self.discriminator.trainable = False
        self.classifier  = self.build_classifier()
        self.classifier.compile(loss='categorical_crossentropy',
                                optimizer='adam',
                                metrics=['acc'])
        self.encoder = self.build_encoder()
        
        target_ipt = Input(shape=self.mil_spec_shape, name='target_ipt')
        intlr_ipt = Input(shape=self.mil_spec_shape, name='intlr_ipt')
        selfpre_ipt = Input(shape=self.mil_spec_shape, name='selfpre_ipt')
        h = self.encoder([target_ipt, intlr_ipt, selfpre_ipt])
        validity = self.discriminator(h)
        emo, gend = self.classifier(h)
        
        self.aec =  Model(inputs=[target_ipt, intlr_ipt, selfpre_ipt], outputs=[emo, gend], name='AEC')
        self.aec.compile(optimizer='adam',
#                     optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), 
                    loss_weights=[1, 1], 
                    loss=['categorical_crossentropy', 'categorical_crossentropy'], 
                    metrics=['acc'])
        self.aec.summary()
        
        
    def build_encoder(self):
        
        def dilated_res_block_proj(x, filters_list, t, strides=1, use_bias=True, name=None):

            out = TimeDistributed(Conv1D(filters=filters_list[0], kernel_size=1, strides=1, 
                                         padding='causal', use_bias=False, 
                                         name='%s_1'%(name)))(x)
            out = Activation('relu', name='%s_1_relu'%(name))(out)

            out = TimeDistributed(Conv1D(filters=filters_list[1], kernel_size=3, strides=1, 
                                         padding='causal', use_bias=False, 
                                         name='%s_2'%(name)))(out)
            out = Activation('relu', name='%s_2_relu'%(name))(out)

            out = TimeDistributed(Conv1D(filters=filters_list[2], kernel_size=1, strides=1, 
                                         padding='causal', dilation_rate=2, use_bias=False, 
                                         name='%s_3'%(name)))(out)

            x = TimeDistributed(Conv1D(filters=filters_list[2], kernel_size=1, strides=1, 
                                       use_bias=False, 
                                       name='%s_proj'%(name)))(x)

            out = Add(name='%s_add'%(name))([x, out])
            out = Activation('relu', name='%s_relu'%(name))(out)    
            if t%5==0:
                out = TimeDistributed(MaxPooling1D(2))(out)
            return out

        def att_hiddenfeat(ipt, iptname): 
            cf = TimeDistributed(Flatten())(ipt)

            spec_btn = TimeDistributed(Dense(512, activation='relu'))(cf)
            spec_gru = Bidirectional(GRU(256, return_sequences=True, activation='sigmoid'))(cf)


            spec_att = Attention()([spec_btn, spec_gru])
            spec_att = Activation('sigmoid')(spec_att)

            specsman_query_value_attention = GlobalAveragePooling1D()(spec_att)
            specsman_query_coding = GlobalAveragePooling1D()(spec_btn)
            specsman_h = multiply([specsman_query_value_attention, specsman_query_coding])
            specsman_h = LeakyReLU(alpha=0.2, name=iptname+'_specsman_h')(specsman_h)


            return specsman_h
        
        target_ipt = Input(shape=self.mil_spec_shape, name='target_ipt')
        intlr_ipt = Input(shape=self.mil_spec_shape, name='intlr_ipt')
        selfpre_ipt = Input(shape=self.mil_spec_shape, name='selfpre_ipt')


        conv_target = target_ipt
        conv_intlr = intlr_ipt
        conv_selfpre = selfpre_ipt

        for _ in range(2):
            conv_target = dilated_res_block_proj(conv_target, [512, 512, 128], t=_, name='target_dilated_'+str(_))
            conv_intlr = dilated_res_block_proj(conv_intlr, [512, 512, 128], t=_, name='intlr_dilated_'+str(_))
            conv_selfpre = dilated_res_block_proj(conv_selfpre, [512, 512, 128], t=_, name='selfpre_dilated_'+str(_))

        target_h = att_hiddenfeat(conv_target, 'target')
        intlr_h = att_hiddenfeat(conv_intlr, 'intlr')
        selfpre_h = att_hiddenfeat(conv_selfpre, 'selfpre')

        att_interact = Softmax()(multiply([intlr_h, selfpre_h]))
        target = multiply([att_interact, target_h])

        # h = Dense(512)(Concatenate(axis=-1)([target, intlr_h, selfpre_h]))
        # target_h = multiply([att_interact, target_h])
        all_info = Concatenate(axis=-1)([target_h, target])
        # h = Concatenate(axis=-1)([selfpre_h, intlr_h, target_h])
        h = Concatenate(axis=-1)([all_info, selfpre_h, intlr_h])
        
        encoder = Model(inputs=[target_ipt, intlr_ipt, selfpre_ipt], outputs=h, name='encoder')
        
        encoder.summary()
        return encoder
        
    def build_classifier(self):
        
        h = Input((2048,), name='hiddendfeat')
        gend = Dense(2, activation='softmax', name='gend')(h)
        emo = Dense(4, activation='softmax', name='emo')(h)
        classifier = Model(h, [emo, gend], name='classifier')
        classifier.summary()
        
        return classifier
        
    def build_discriminator(self):
        model = Sequential()

        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))

        encoded_repr = Input(shape=(self.latent_dim, ))
        validity = model(encoded_repr)
        discriminator = Model(encoded_repr, validity, name='discriminator')
        discriminator.summary()
        
        return discriminator
    
    
    def train(self, epochs, batch_size, data):
        [mi_tra_specs, mi_tra_intlter, mi_tra_pre_self, tra_emos, tra_gends,
        mi_tes_specs, mi_tes_intlter, mi_tes_pre_self, tes_emos, tes_gends] = data
        
        acc_cla_max = 0.0
        
        file_path_root = './results/'
        model_file = file_path_root+'aec'

        for epoch in range(epochs):
            print('---- epoch %d ----'%(epoch))
            iters = int(len(tra_specs)/batch_size)
#             iters = 10
            valid = np.ones((batch_size, 1))
            fake = np.zeros((batch_size, 1))
            
            for it in range(iters):
                idx = np.random.randint(0, mi_tra_specs.shape[0], batch_size)
                target_ipt = mi_tra_specs[idx]
                intlr_ipt = mi_tra_intlter[idx]
                selfpre_ipt = mi_tra_pre_self[idx]

                emos = tra_emos[idx]
                gends = tra_gends[idx]
                
                latent_fake = self.encoder.predict([target_ipt, intlr_ipt, selfpre_ipt])
                latent_real = np.random.normal(size=(batch_size, self.latent_dim))
                
                d_loss_real = self.discriminator.train_on_batch(latent_real, valid)
                d_loss_fake = self.discriminator.train_on_batch(latent_fake, fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                print("Train Discriminator:%d/%d | loss %.2f  acc %.2f"%(it+1, iters, d_loss[0], d_loss[1]), end='\r')
                
            print("Train Discriminator:%d/%d | loss %.2f  acc %.2f"%(it+1, iters, d_loss[0], d_loss[1]))
            
            
            for it in range(iters):
                idx = np.random.randint(0, mi_tra_specs.shape[0], batch_size)
                target_ipt = mi_tra_specs[idx]
                intlr_ipt = mi_tra_intlter[idx]
                selfpre_ipt = mi_tra_pre_self[idx]

                emos = tra_emos[idx]
                gends = tra_gends[idx]

                g_loss = self.aec.train_on_batch([target_ipt, intlr_ipt, selfpre_ipt], 
                                                  [emos, gends])
                
                print("Train AEC:%d/%d | loss:%.4f  emo_loss:%.4f  gend_loss:%.4f  emo_acc:%.4f  gend_acc:%.4f"%(it+1, iters, g_loss[0], g_loss[1], g_loss[2],
                                                                                              g_loss[3]*100, g_loss[4]*100), end='\r')       
            print("Train AEC:%d/%d | loss:%.4f  emo_loss:%.4f  gend_loss:%.4f  emo_acc:%.4f  gend_acc:%.4f"%(it+1, iters, g_loss[0], g_loss[1], g_loss[2],
                                                                                              g_loss[3]*100, g_loss[4]*100))   
            
            
            emos_pred, _ = self.aec.predict([mi_tes_specs, mi_tes_intlter, mi_tes_pre_self])
            predicted_test_labels = emos_pred.argmax(axis=1)
            numeric_test_labels = np.array(tes_emos).argmax(axis=1)
            
            eval_r_aec = classification_report(numeric_test_labels, predicted_test_labels, 
                                           target_names = ['joy', 'sad', 'neu', 'ang'], 
                                           digits=4, output_dict=True)
            
            acc_cla = eval_r_aec['accuracy']
            print('classifier acc', acc_cla)
            print('classifier \n', classification_report(numeric_test_labels, predicted_test_labels, 
                                                   target_names = ['joy', 'sad', 'neu', 'ang'], 
                                                   digits=4))
            if acc_cla>acc_cla_max:
                print("--------------------")
                print("acc improved from %.4f to %.4f"%(acc_cla_max, acc_cla))
                print("--------------------")
                acc_cla_max = acc_cla
                save_model(self.aec, model_file)
                
                report_filename = file_path_root+'cla.txt'
                with open(report_filename, 'w', encoding='utf-8') as f:
                    print(classification_report(numeric_test_labels, predicted_test_labels, 
                                                target_names = ['joy', 'sad', 'neu', 'ang'], 
                                                digits=4), file=f)
                file_path = file_path_root+'cla_'
                confusion_matrix_cal(numeric_test_labels, predicted_test_labels, file_path)
            else:
                print("***************")
                print("acc not improved from %.4f"%acc_cla_max)
                print("***************")
            



