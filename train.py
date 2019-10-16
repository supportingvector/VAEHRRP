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
import utils
import test_data
      
def train_with_label(EPOCH,BATCH_SIZE,last_checkpoint,train_loader,vae_encoder,vae_decoder,classifier):
      epoch_loss=[]

      optimizer_encoder =  optim.Adam(vae_encoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)   
      optimizer_encoder_prior_mu =  optim.Adam([vae_encoder.prior_mu], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)   
      optimizer_encoder_prior_logvar =  optim.Adam([vae_encoder.prior_logvar], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)   

      optimizer_decoder =  optim.Adam(vae_decoder.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)  
      optimizer_encoder_kkt_param =  optim.Adam([vae_encoder.kkt_param], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)   
 
      optimizer_classifier=optim.Adam(classifier.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-8)   
      criterion = torch.nn.CrossEntropyLoss()
      
      for epoch in range(EPOCH):
        total_loss = 0
        for i in range(int((52000+52000+36000)/BATCH_SIZE)):
            x,label_onehot,average_x,label=next(train_loader)
            
            x= torch.tensor(x).type(torch.FloatTensor).cuda()
            average_x= torch.tensor(average_x).type(torch.FloatTensor).cuda()
            label_onehot= torch.tensor(label_onehot).type(torch.FloatTensor).cuda()
            label=torch.tensor(label).type(torch.LongTensor).cuda()
            
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_encoder_prior_mu.zero_grad()
            optimizer_encoder_prior_logvar.zero_grad()
            optimizer_encoder_kkt_param.zero_grad()
            optimizer_classifier.zero_grad()
            
            z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,prior_mu,prior_logvar = vae_encoder.forward(x,label_onehot)
            
            
            recon_x_mu,recon_x_logvar = vae_decoder.forward(z_g,z_d)
            classifier_label=classifier.forward(z_d)
            
            
#            loss,idendity_loss,average_loss,KLD,JS = utils.loss_func(average_recon_x_mu,average_recon_x_sigma2,average_x,
#                                                            recon_x_mu,recon_x_sigma2, 
#                                                            x, mu, logvar,label_onehot,prior_mu,
#                                                            BATCH_SIZE)
            idendity_loss=utils.reconstruct_loss(x,recon_x_mu,recon_x_logvar)
            
            KLD_g=utils.KLD_loss_general(logvar_g,label_onehot,mu_g)

            KLD_d=utils.KLD_loss(logvar_d,label_onehot,prior_mu,prior_logvar,mu_d)
            
            H=torch.nn.functional.relu(vae_encoder.kkt_param)*(utils.H_loss(prior_mu,prior_logvar)-1*len(x))
            
            classifier_loss=utils.classifier_loss(criterion,label,classifier_label)
            
            
            #loss=-((idendity_loss)-KLD_g-KLD_d+H)+classifier_loss
            loss=-(idendity_loss-KLD_g-KLD_d)-H+classifier_loss
            loss.backward()
            total_loss += idendity_loss.data[0]

            optimizer_encoder.step()     
            
            optimizer_decoder.step() 
            if i%10==1:
                optimizer_encoder_prior_mu.step() 
                optimizer_encoder_prior_logvar.step()
            optimizer_encoder_kkt_param.step()
            optimizer_classifier.step()
            print('Train with label Epoch:{} -- [{}/{} ({:.0f}%)] -- Loss:{:.6f}'.format(
                  epoch, i*len(x), 52000+52000+36000, 
                  1./28*i*50/BATCH_SIZE, loss.data[0]/len(x)))         
            print(' idendity_loss:{:.6f}  KLD:{:.6f} H:{:.6f}'.format(idendity_loss.data[0]/len(x),KLD_d.data[0]/len(x),H.data[0]/len(x)))
        print('====> Epoch: {} Average loss: {:.4f}'.format(
                  epoch, total_loss / (52000+52000+36000))) 
        epoch_loss.append(total_loss / (52000+52000+36000))
        if epoch%10==0:
            '''存原来的训练文件'''    
            torch.save(vae_encoder.state_dict(), 'save\\vae_encoder'+str(last_checkpoint+epoch)+'.pkl')
            torch.save(vae_decoder.state_dict(), 'save\\vae_decoder'+str(last_checkpoint+epoch)+'.pkl')
            torch.save(classifier.state_dict(), 'save\\classifier'+str(last_checkpoint+epoch)+'.pkl')
            
            z_prior_mu = vae_encoder.prior_mu.detach().cpu().numpy()
            z_prior_logvar = vae_encoder.prior_logvar.detach().cpu().numpy()
            kkt_param = vae_encoder.kkt_param.detach().cpu().numpy()
            np.save('save\\z_prior_mu'+str(last_checkpoint+epoch)+'.npy',z_prior_mu)
            np.save('save\\z_prior_logvar'+str(last_checkpoint+epoch)+'.npy',z_prior_logvar)
            np.save('save\\kkt_param'+str(last_checkpoint+epoch)+'.npy',kkt_param)
            
            test_data.test_z1(BATCH_SIZE,vae_encoder)
            test_data.test_z2(BATCH_SIZE,vae_encoder)
            test_data.test_z3(BATCH_SIZE,vae_encoder)
            z1test=np.load('save\\z1t.npy').tolist()
            z2test=np.load('save\\z2t.npy').tolist()
            z3test=np.load('save\\z3t.npy').tolist()
            ztest=z1test+z2test+z3test
            ztest=torch.tensor(np.array(ztest)).type(torch.FloatTensor).cuda().view(5200,-1)
            
            yt1=[0 for i in range(2000)]
            yt2=[1 for i in range(2000)]
            yt3=[2 for i in range(1200)]    
            ytest=yt1+yt2+yt3
            y_label=torch.tensor(np.array(ytest)).type(torch.LongTensor).cuda()
            outputs = classifier(ztest)
            _, predicted = torch.max(outputs.data, 1)
            correct=0
            correct += (predicted == y_label).sum().item()
            print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / 5200))

      return epoch_loss
test_data.test_z1(BATCH_SIZE,vae_encoder)
test_data.test_z2(BATCH_SIZE,vae_encoder)
test_data.test_z3(BATCH_SIZE,vae_encoder)
z1test=np.load('save\\z1t.npy').tolist()
z2test=np.load('save\\z2t.npy').tolist()
z3test=np.load('save\\z3t.npy').tolist()
ztest=z1test+z2test+z3test
ztest=torch.tensor(np.array(ztest)).type(torch.FloatTensor).cuda().view(5200,-1)

yt1=[0 for i in range(2000)]
yt2=[1 for i in range(2000)]
yt3=[2 for i in range(1200)]    
ytest=yt1+yt2+yt3
y_label=torch.tensor(np.array(ytest)).type(torch.LongTensor).cuda()
outputs = classifier(ztest)
_, predicted = torch.max(outputs.data, 1)
correct=0
correct += (predicted == y_label).sum().item()
print('Accuracy of the network on the 5200 test images: %d %%' % (100 * correct / 5200))
 
              






    
    
    
      