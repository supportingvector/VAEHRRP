#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:48:03 2018

@author: xms
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import hrrp_dataloader
from torch.nn import init
import numpy as np
from matplotlib import pyplot as plt

import train
import model
import test_data
import utils
import test_f
'''
TODO:  decoder mlp增加，encoder可以先用训练好的参数
'''
                    
EPOCH = 11
BATCH_SIZE = 200
LATENT_CODE_g = 30#40 general 
LATENT_CODE_d = 20
log_interval = 10
last_checkpoint=0

droupout_rate=0
vae_encoder = model.VAE_encoder(droupout_rate,LATENT_CODE_g,LATENT_CODE_d).cuda()
vae_decoder = model.VAE_decoder(LATENT_CODE_g,LATENT_CODE_d).cuda()
classifier=model.Classifier(LATENT_CODE_d).cuda()

'''训练'''

'''读取原来的训练文件'''

if last_checkpoint!=0:
    vae_encoder.load_state_dict(torch.load('save\\vae_encoder'+str(last_checkpoint)+'.pkl'))
    vae_decoder.load_state_dict(torch.load('save\\vae_decoder'+str(last_checkpoint)+'.pkl'))
    classifier.load_state_dict(torch.load('save\\classifier'+str(last_checkpoint)+'.pkl'))    
    z_prior_mu=np.load('save\\z_prior_mu'+str(last_checkpoint)+'.npy')
    z_prior_logvar=np.load('save\\z_prior_logvar'+str(last_checkpoint)+'.npy')
    kkt_param=np.load('save\\kkt_param'+str(last_checkpoint)+'.npy')
    vae_encoder.prior_mu=Variable(torch.tensor(z_prior_mu).type(torch.FloatTensor).cuda(), requires_grad=True)
    vae_encoder.prior_logvar=Variable(torch.tensor(z_prior_logvar).type(torch.FloatTensor).cuda(), requires_grad=True)
    
    vae_encoder.kkt_param=Variable(torch.tensor(kkt_param).type(torch.FloatTensor).cuda(), requires_grad=True)

print('train start')
train_loader = hrrp_dataloader.train_dataloader_mult_prior(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
epoch_loss=train.train_with_label(EPOCH,BATCH_SIZE,last_checkpoint,train_loader,vae_encoder,vae_decoder,classifier)


'''测试'''

load_checkpoint=100
droupout_rate=0
#z_prior_mu=np.load('save\\z_prior_mu'+str(load_checkpoint)+'.npy')
#z_prior_logvar=np.load('save\\z_prior_logvar'+str(last_checkpoint)+'.npy')
#vae_encoder = model.VAE_encoder(droupout_rate,LATENT_CODE_g,LATENT_CODE_d).cuda().eval()
#vae_decoder = model.VAE_decoder(LATENT_CODE_g,LATENT_CODE_d).eval()
#vae_encoder.load_state_dict(torch.load('save\\vae_encoder'+str(load_checkpoint)+'.pkl'))
#vae_decoder.load_state_dict(torch.load('save\\vae_decoder'+str(load_checkpoint)+'.pkl'))
#classifier.load_state_dict(torch.load('save\\classifier'+str(load_checkpoint)+'.pkl'))

#test_data.train_z1(BATCH_SIZE,vae_encoder)
#test_data.train_z2(BATCH_SIZE,vae_encoder)
#test_data.train_z3(BATCH_SIZE,vae_encoder)
#test_data.test_z1(BATCH_SIZE,vae_encoder)
#test_data.test_z2(BATCH_SIZE,vae_encoder)
#test_data.test_z3(BATCH_SIZE,vae_encode)
#test_data.show_z1(BATCH_SIZE,vae_encoder,vae_decoder)
#test_data.show_z2(BATCH_SIZE,vae_encoder,vae_decoder)
#test_data.show_z3(BATCH_SIZE,vae_encoder,vae_decoder)
#test_data.show_zt1(BATCH_SIZE,vae_encoder,vae_decoder)
#test_data.show_zt2(BATCH_SIZE,vae_encoder,vae_decoder)
#test_data.show_zt3(BATCH_SIZE,vae_encoder,vae_decoder)
#test_f.test_accuracy()





    
    
    
      