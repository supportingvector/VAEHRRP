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

'''
TODO:  decoder mlp增加，encoder可以先用训练好的参数
'''
class VAE_encoder(nn.Module):
      def __init__(self,droupout_rate,LATENT_CODE_g,LATENT_CODE_d):
            super(VAE_encoder, self).__init__()    
            '''cnn part'''
            self.encoder = nn.Sequential(
                  nn.Conv1d(1, 32, kernel_size=9, stride=1, padding=4),
                  nn.BatchNorm1d(32),
                  nn.LeakyReLU(0.2, inplace=True),
                  #complex_option.Complex_LeakyReLU(0.2),
                  nn.MaxPool1d(2),
                  
                  nn.Conv1d(32, 32, kernel_size=9, stride=1, padding=4),
                  nn.BatchNorm1d(32),
                  nn.LeakyReLU(0.2, inplace=True),
                  #complex_option.Complex_LeakyReLU(0.2),
                  nn.MaxPool1d(2)                   
                  )        
            '''fc part'''                
            self.fc11 = nn.Sequential(nn.Linear(32*64, 500),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(500, 250),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(250, LATENT_CODE_g),
#                                      torch.nn.Dropout(droupout_rate),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      )
            
            self.fc12 = nn.Sequential(nn.Linear(32*64, 500),
#                                      torch.nn.Dropout(droupout_rate),
                                      nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(500, 250),
                                      nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(250, LATENT_CODE_g),
                                      nn.LeakyReLU(0.2, inplace=True)
                                      )
            self.fc21 = nn.Sequential(nn.Linear(32*64, 500),
                                     nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(500, 250),
                                     nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(250, LATENT_CODE_d),
#                                      torch.nn.Dropout(droupout_rate),
                                      nn.LeakyReLU(0.2, inplace=True),
                                      )
            
            self.fc22 = nn.Sequential(nn.Linear(32*64, 500),
#                                      torch.nn.Dropout(droupout_rate),
                                      nn.LeakyReLU(0.2, inplace=True),
                                     nn.Linear(500, 250),
                                      nn.LeakyReLU(0.2, inplace=True),

                                     nn.Linear(250, LATENT_CODE_d),
                                      nn.LeakyReLU(0.2, inplace=True)
                                      )
            
            self.prior_mu=Variable(torch.randn(1,3,LATENT_CODE_d).type(torch.FloatTensor).cuda(), requires_grad=True)
            self.kkt_param=Variable(torch.randn(1).type(torch.FloatTensor).cuda(), requires_grad=True)
            self.prior_logvar=Variable(torch.randn(1,3,LATENT_CODE_d).type(torch.FloatTensor).cuda(), requires_grad=True)
        
      def reparameterize(self, mu, logvar):
            eps = Variable(torch.randn(mu.size(0), mu.size(1))).cuda()
            z = mu + eps * torch.exp(logvar/2)                       
            return z
      
      def forward(self,x,label):
            '''
            输入值和对应的类别名，类别监督信息在全连接层融合，返回z，mu, logvar
            '''
            out1,out2= self.encoder(x),self.encoder(x)            
            mu_g = self.fc11(out1.view(out1.size(0),-1))     # batch_s, latent
            logvar_g=self.fc12(out1.view(out1.size(0),-1))
            z_g = self.reparameterize(mu_g, logvar_g)      # batch_s, latent          
            
            mu_d = self.fc21(out2.view(out1.size(0),-1))     # batch_s, latent
            logvar_d=self.fc22(out1.view(out1.size(0),-1))
            z_d = self.reparameterize(mu_d, logvar_d)      # batch_s, latent    

            return z_g,z_d, mu_g,mu_d, logvar_g,logvar_d,self.prior_mu,self.prior_logvar

#class VAE_decoder(nn.Module):
#      def __init__(self,LATENT_CODE_NUM):
#            super(VAE_decoder, self).__init__()               
#          
#            self.decoder_mu = nn.Sequential(                
#                  complex_option.Complex_ConvTranspose1d(2, 32, kernel_size=4, stride=2, padding=1),
#                  nn.LeakyReLU(0.2, inplace=True),
#                  #complex_option.Complex_LeakyReLU(0.2),
#                  complex_option.Complex_ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
#                  nn.LeakyReLU(0.2, inplace=True)
#                  )
#            self.decoder_sigma2 = nn.Sequential(                
#                  complex_option.Complex_ConvTranspose1d(2, 32, kernel_size=4, stride=2, padding=1),
#                  nn.LeakyReLU(0.2, inplace=True),
#                  #complex_option.Complex_LeakyReLU(0.2),
#                  complex_option.Complex_ConvTranspose1d(32, 2, kernel_size=4, stride=2, padding=1),
#                  nn.LeakyReLU(0.2, inplace=True)
#                  )
#            self.fc2 = nn.Sequential(
#                                    complex_fc.Complex_Linear(LATENT_CODE_NUM, 80),
#                                     nn.LeakyReLU(0.2, inplace=True),  
#                                     complex_fc.Complex_Linear(80, 64*2),
#                                     nn.LeakyReLU(0.2, inplace=True) 
#                                     )          
#      
#      def forward(self, z):   
#             z_reshape = self.fc2(z).view(z.size(0), 2, 64)    # batch_s, 8, 7, 7
#             recon_x_mu=self.decoder_mu(z_reshape)
#             recon_x_sigma2=self.decoder_sigma2(z_reshape)
#             
#             return recon_x_mu,recon_x_sigma2

class VAE_decoder(nn.Module):
      def __init__(self,LATENT_CODE_g,LATENT_CODE_d):
            super(VAE_decoder, self).__init__()               
          
            self.decoder_mu = nn.Sequential(                
                  nn.ConvTranspose1d(1, 32, kernel_size=4, stride=2, padding=1),
                  #complex_option.Complex_LeakyReLU(0.2),
                  nn.LeakyReLU(0.2,inplace=True),
                  nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)
                  )
            self.decoder_logvar = nn.Sequential(                
                  nn.ConvTranspose1d(1, 32, kernel_size=4, stride=2, padding=1),
                  #complex_option.Complex_LeakyReLU(0.2),
                  nn.LeakyReLU(0.2,inplace=True),
                  nn.ConvTranspose1d(32, 1, kernel_size=4, stride=2, padding=1),
                  nn.LeakyReLU(0.2, inplace=True)
                  )                    
            self.fc2 = nn.Sequential(
                                     nn.Linear(LATENT_CODE_g+LATENT_CODE_d,64),
                                     nn.LeakyReLU(0.2, inplace=True),  
                                     nn.Linear(64, 64),
                                     nn.LeakyReLU(0.2, inplace=True)                
                                     ) 
                                         
      
      def forward(self, z_g,z_d):   
             z=torch.cat([z_g,z_d],1)
             z_reshape = self.fc2(z).view(z.size(0), 1, 64)    # batch_s, 8, 7, 7
             
             # average_recon_x_mu=torch.norm(recon_x,p=2,dim=1)
             average_recon_x_mu=self.decoder_mu(z_reshape)
             average_recon_x_logvar=self.decoder_logvar(z_reshape)
             
             return average_recon_x_mu,average_recon_x_logvar
        
class Classifier(nn.Module):
    def __init__(self,LATENT_CODE_d):
        super(Classifier, self).__init__()  
        self.fc = nn.Sequential(
                                     nn.Linear(LATENT_CODE_d,64),
                                     nn.LeakyReLU(0.2, inplace=True),  
                                     nn.Linear(64, 64),
                                     nn.LeakyReLU(0.2, inplace=True),   
                                     nn.Linear(64,3)
                                     )
    def forward(self,z_d):
        z=torch.cat([z_d],1)
        out = self.fc(z)
        return out
    
           

              
        






    
    
    
      