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
'''测试用的z生成'''

def train_z1(BATCH_SIZE,vae_encoder):
#    result=torch.zeros([0,0])
   train_loader = hrrp_dataloader.train_data_generator1(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
   result=[]
   for i in range(int((52000)/BATCH_SIZE)):
   ##52000,52000,36000,2000,2000,1200
           
       x,label=next(train_loader)            
       x= torch.tensor(x).type(torch.FloatTensor).cuda()
       label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
       z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
       z_d=z_d.cpu()
       z_g=z_g.cpu()
       result.append(z_d.data.numpy())
       #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
   X=tuple(result)
   X=np.concatenate(X,axis=0)
   plt.hist(X, bins=200, histtype="stepfilled", alpha=.8)
   plt.title(r'Histogram of Class 1 train data') 
   plt.show()
   np.save('save\\z1.npy',X)  
   return result           

def train_z2(BATCH_SIZE,vae_encoder):
#    result=torch.zeros([0,0])
   train_loader = hrrp_dataloader.train_data_generator2(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
   result=[]
   for i in range(int((52000)/BATCH_SIZE)):
   ##52000,52000,36000,2000,2000,1200
           
       x,label=next(train_loader)            
       x= torch.tensor(x).type(torch.FloatTensor).cuda()
       label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
       z_g,z, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar= vae_encoder.forward(x,label)
       z=z.cpu()
       result.append(z.data.numpy())
       #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
   X=tuple(result)
   X=np.concatenate(X,axis=0)
   plt.hist(X, bins=200, histtype="stepfilled", alpha=.8)
   plt.title(r'Histogram of Class 2 train data') 
   plt.show()
   np.save('save\\z2.npy',X)        
   return result           

def train_z3(BATCH_SIZE,vae_encoder):
#    result=torch.zeros([0,0])
   train_loader = hrrp_dataloader.train_data_generator3(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
   result=[]
   for i in range(int((36000)/BATCH_SIZE)):
   ##52000,52000,36000,2000,2000,1200
           
       x,label=next(train_loader)            
       x= torch.tensor(x).type(torch.FloatTensor).cuda()
       label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
       z_g,z, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
       z=z.cpu()
       result.append(z.data.numpy())
       #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
   X=tuple(result)
   X=np.concatenate(X,axis=0)
   plt.hist(X, bins=200, histtype="stepfilled", alpha=.8)
   plt.title(r'Histogram of Class 3 train data') 
   plt.show()
   np.save('save\\z3.npy',X)
   return result            

def test_z1(BATCH_SIZE,vae_encoder):
    train_loader = hrrp_dataloader.train_data_generator_t1(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
#    result=torch.zeros([0,0])
    result=[]
    for i in range(int((2000)/BATCH_SIZE)):
    ##52000,52000,36000,2000,2000,1200
            
        x,label=next(train_loader)            
        x= torch.tensor(x).type(torch.FloatTensor).cuda()
        label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
        z_g,z, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
        z=z.cpu()
        result.append(z.data.numpy())
        #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
    X=tuple(result)
    X=np.concatenate(X,axis=0)
    plt.hist(X, bins=80, histtype="stepfilled", alpha=.8)
    plt.title(r'Histogram of Class 1 test data') 
    plt.show()
    np.save('save\\z1t.npy',X)
    return result           

def test_z2(BATCH_SIZE,vae_encoder):
#    result=torch.zeros([0,0])
    train_loader = hrrp_dataloader.train_data_generator_t2(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]
    for i in range(int((2000)/BATCH_SIZE)):
    ##52000,52000,36000,2000,2000,1200
            
        x,label=next(train_loader)            
        x= torch.tensor(x).type(torch.FloatTensor).cuda()
        label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
        z_g,z, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
        z=z.cpu()
        result.append(z.data.numpy())
        #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
    X=tuple(result)
    X=np.concatenate(X,axis=0)
    plt.hist(X, bins=80, histtype="stepfilled", alpha=.8)
    plt.title(r'Histogram of Class 2 test data') 
    plt.show()
    np.save('save\\z2t.npy',X)
    return result           

def test_z3(BATCH_SIZE,vae_encoder):
#    result=torch.zeros([0,0])
    train_loader = hrrp_dataloader.train_data_generator_t3(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)

    result=[]
    for i in range(int((1200)/BATCH_SIZE)):
    ##52000,52000,36000,2000,2000,1200
            
        x,label=next(train_loader)            
        x= torch.tensor(x).type(torch.FloatTensor).cuda()
        label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
        z_g,z, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
        z=z.cpu()
        result.append(z.data.numpy())
        #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))
    X=tuple(result)
    X=np.concatenate(X,axis=0)
    plt.hist(X, bins=80, histtype="stepfilled", alpha=.8)
    plt.title(r'Histogram of Class 3 test data') 
    plt.show()
    np.save('save\\z3t.npy',X)
    #
    return result   

def show_z1(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator1(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()
    
def show_z2(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator2(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()
    
def show_z3(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator3(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()
    
def show_zt1(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator_t1(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()
    
def show_zt2(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator_t2(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()
    
def show_zt3(BATCH_SIZE,vae_encoder,vae_decoder):
#    result=torch.zeros([0,0])
    index=np.array(range(0,256))
    train_loader = hrrp_dataloader.train_data_generator_t3(file_path=r'D:\科研\VAEHRRP',batch_size=BATCH_SIZE)
    result=[]           
    x,label=next(train_loader)
    plt.plot(index,x[0].reshape(256))
    x= torch.tensor(x).type(torch.FloatTensor).cuda()
    label = torch.zeros(BATCH_SIZE,3).type(torch.FloatTensor).cuda()            
    z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label)
    z_d=z_d.cpu()
    z_g=z_g.cpu()
    result.append(z_d.data.numpy())
    #print('test  -- [{}/{} ({:.0f}%)]'.format(i*len(x), 2000, 1./28*i*50/BATCH_SIZE))     
    mu,logvar=vae_decoder.forward(z_g,z_d)
    plt.plot(index,mu.data.numpy()[0].reshape(256))
    plt.show()




    
    
    
      