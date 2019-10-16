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
import scipy.io as scio
import model
             
def H_divergence(mu1,mu2,sigma1=1,sigma2=1):
    '''
    返回两个高斯分布的JS散度
    TODO:JS
    '''
    H_2=1-pow(2*sigma1*sigma2/(pow(sigma1,2)+pow(sigma2,2)),0.5)*torch.exp(-1/4*(mu1-mu2).pow(2)/(pow(sigma1,2)+pow(sigma2,2)))
    return H_2
 
def loss_func(average_recon_x_mu,average_recon_x_sigma2,average_x,recon_x_mu,recon_x_sigma2, x, mu, logvar,label_onehot,prior_mu,BATCH_SIZE):      
#      idendity_loss = torch.sum(torch.pow((recon_x_mu- x),2)/recon_x_sigma2/2)
#      average_loss = torch.sum(torch.pow((average_recon_x_mu- average_x),2)/average_recon_x_sigma2/2)
#     
      sigma_prior=0.1
      idendity_loss = torch.sum(torch.pow((recon_x_mu- x),2)/(2*0.0001))
      #idendity_loss = torch.sum(torch.pow((recon_x_mu- x),2)/2)
      average_loss = torch.sum(torch.pow((average_recon_x_mu- average_x),2)/(2*0.0001))
      #print(torch.max(label_onehot*prior_mu,1)[0][0])
#      KLD = -0.5 * torch.sum(1 + logvar - (mu-torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
#      *prior_mu,1)[0].view(logvar.shape[0],logvar.shape[1])).pow(2) - logvar.exp())
      mu_muprior=mu-torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
      *prior_mu,1)[0].view(logvar.shape[0],logvar.shape[1])
      
      KLD = -0.5 * torch.sum(1 + logvar - (mu_muprior.pow(2) +logvar.exp()*1)/(sigma_prior*sigma_prior))
            
      H1=H_divergence(prior_mu[:,0],prior_mu[:,1],sigma1=sigma_prior,sigma2=sigma_prior)
      H2=H_divergence(prior_mu[:,0],prior_mu[:,2],sigma1=sigma_prior,sigma2=sigma_prior)
      H3=H_divergence(prior_mu[:,2],prior_mu[:,1],sigma1=sigma_prior,sigma2=sigma_prior)
      H_sum=0.0001*torch.sum(1*(H1+H2+H3))
      
      #return idendity_loss+average_loss+KLD-H_sum,idendity_loss,average_loss,KLD,H_sum
      return idendity_loss+average_loss,idendity_loss,average_loss,KLD,H_sum

def reconstruct_loss(x,recon_x_mu,recon_x_logvar):
    idendity_loss = torch.sum(-torch.pow((recon_x_mu- x),2)/(2*torch.exp(recon_x_logvar))-0.5*recon_x_logvar)
    return idendity_loss
    
def reconstruct_loss_a(average_x,average_recon_x_mu):
    average_loss = torch.sum(torch.pow((average_recon_x_mu- average_x),2)/(2*0.0001))
    return average_loss
    
def KLD_loss_general(logvar,label_onehot,mu):
    KLD = -0.5 * torch.sum(1 + logvar - torch.pow(mu,2)-logvar.exp())
    return KLD
    
def KLD_loss(logvar,label_onehot,prior_mu,prior_logvar,mu,latent_num_d=20):
#    print(torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
#      *prior_mu,1)[0].shape)
#    sigma_prior=prior_logvar
#    a=torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
#      *prior_mu,1)[0]
#    print(a.shape)
    mu_muprior=mu-torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
      *prior_mu,1)[0].view(logvar.shape[0],latent_num_d)
    prior_logvar=torch.max(label_onehot.view(label_onehot.shape[0],label_onehot.shape[1],1)
      *prior_logvar,1)[0].view(logvar.shape[0],latent_num_d)
    KLD = -0.5 * torch.sum(1 + logvar-prior_logvar - (mu_muprior.pow(2) +logvar.exp())/prior_logvar.exp())
    
    return KLD
    
def H_loss(prior_mu,prior_logvar):
    H1=H_divergence(prior_mu[:,0],prior_mu[:,1],sigma1=prior_logvar[:,0].exp(),sigma2=prior_logvar[:,1].exp())
    H2=H_divergence(prior_mu[:,0],prior_mu[:,2],sigma1=prior_logvar[:,0].exp(),sigma2=prior_logvar[:,2].exp())
    H3=H_divergence(prior_mu[:,2],prior_mu[:,1],sigma1=prior_logvar[:,2].exp(),sigma2=prior_logvar[:,1].exp())
    H_sum=torch.sum(1*(H1+H2+H3))
    return H_sum
    
def classifier_loss(criterion,label,classifier_label):
#    print(label.shape)
#    print(classifier_label.shape)
    return criterion(classifier_label,label)
      
def load_visual_data_c1(file_path):
    index=250
    dataFile_real1 = file_path+r'\test2\test_data.mat'
    real_dict = scio.loadmat(dataFile_real1)
    data_real1=real_dict['test_data']
    real=data_real1[index,:]
#
#    dataFile_average =file_path+ r'\train2\train_average.mat'
#    train_average_dict = scio.loadmat(dataFile_average)
#    train_average=train_average_dict['train_average']
#    train_average=train_average[index,:]    

    
    real=np.expand_dims(real, axis=0)
#    train_average=np.expand_dims(train_average, axis=0) 

    real=np.expand_dims(real, axis=1)
#    train_average=np.expand_dims(train_average, axis=1) 
    
    return real
    
    
def load_visual_data_c2(file_path):
    index=250
    dataFile_real1 = file_path+r'\test2\test_data2.mat'
    real_dict = scio.loadmat(dataFile_real1)
    data_real1=real_dict['test_data2']
    real=data_real1[index,:]
#
#    dataFile_average =file_path+ r'\train2\train_average.mat'
#    train_average_dict = scio.loadmat(dataFile_average)
#    train_average=train_average_dict['train_average']
#    train_average=train_average[index,:]    

    
    real=np.expand_dims(real, axis=0)
#    train_average=np.expand_dims(train_average, axis=0) 

    real=np.expand_dims(real, axis=1)
#    train_average=np.expand_dims(train_average, axis=1) 
    
    return real
    
    
def load_visual_data_c3(file_path):
    index=250
    dataFile_real1 = file_path+r'\test2\test_data3.mat'
    real_dict = scio.loadmat(dataFile_real1)
    data_real1=real_dict['test_data3']
    real=data_real1[index,:]
#
#    dataFile_average =file_path+ r'\train2\train_average.mat'
#    train_average_dict = scio.loadmat(dataFile_average)
#    train_average=train_average_dict['train_average']
#    train_average=train_average[index,:]    

    
    real=np.expand_dims(real, axis=0)
#    train_average=np.expand_dims(train_average, axis=0) 

    real=np.expand_dims(real, axis=1)
#    train_average=np.expand_dims(train_average, axis=1) 
    
    return real

def visualize(file_path,vae_encoder,vae_decoder):
    '''
    重构可视化
    '''
    train_data_1=load_visual_data_c1(file_path)
    train_data_1= torch.tensor(train_data_1).type(torch.FloatTensor).cuda()
    
    
    re_train_data_1=vae_decoder(vae_encoder.forward(train_data_1,0)[0])
    re_train_data_1=re_train_data_1[0].cpu().detach().numpy()
    
    plt.xlim(0,256)
    plt.ylim(0, 1)
    
    plt.plot(train_data_1[0,0,:],color='green',label = "originial",linestyle=":")
    plt.plot(re_train_data_1[0,0,:],color='gray',label = "reconstruct")
#    plt.legend(loc='upper left')
    plt.show()
    
    train_data_1=load_visual_data_c2(file_path)
    train_data_1= torch.tensor(train_data_1).type(torch.FloatTensor).cuda()
    
    
    re_train_data_1=vae_decoder(vae_encoder.forward(train_data_1,0)[0])
    re_train_data_1=re_train_data_1[0].cpu().detach().numpy()
    
    plt.xlim(0,256)
    plt.ylim(0, 1)
    
    plt.plot(train_data_1[0,0,:],color='green',label = "originial",linestyle=":")
    plt.plot(re_train_data_1[0,0,:],color='gray',label = "reconstruct")
#    plt.legend(loc='upper left')
    plt.show()
    
    train_data_1=load_visual_data_c3(file_path)
    train_data_1= torch.tensor(train_data_1).type(torch.FloatTensor).cuda()
    
    
    re_train_data_1=vae_decoder(vae_encoder.forward(train_data_1,0)[0])
    re_train_data_1=re_train_data_1[0].cpu().detach().numpy()
    
    plt.xlim(0,256)
    plt.ylim(0, 1)
    
    plt.plot(train_data_1[0,0,:],color='green',label = "originial",linestyle=":")
    plt.plot(re_train_data_1[0,0,:],color='gray',label = "reconstruct")
#    plt.legend(loc='upper left')
    plt.show()
    return train_data_1,re_train_data_1
    
    

    
    
    
    

#load_checkpoint=50
#vae_encoder = model.VAE_encoder(droupout_rate=0,LATENT_CODE_NUM=50).cuda().eval()    
#vae_encoder.load_state_dict(torch.load('vae_encoder'+str(load_checkpoint)+'.pkl'))
#   
#vae_decoder = model.VAE_decoder(LATENT_CODE_NUM=50).cuda().eval()    
#vae_decoder.load_state_dict(torch.load('vae_decoder'+str(load_checkpoint)+'.pkl'))
## 
# 
#train_data_1,re_train_data_1=visualize(r'D:\科研\VAEHRRP',vae_encoder,vae_decoder)
#
##


    
    
    
      