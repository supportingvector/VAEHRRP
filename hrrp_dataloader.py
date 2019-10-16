# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 10:31:22 2018

@author: xumingsheng
"""
import scipy.io as scio
import numpy as np
import keras
import torch
			
	
class train_dataloader():
    '''
    TODO: 打乱测试
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            return_data_average = self.data_average[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label,return_data_average
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            return_data_average = self.data_average[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label,return_data_average     
        
    
    def load_train(self,file_path):
            '''第一类数据'''  
            dataFile_1 = file_path+r'\train2\train_data.mat'
            dataFile_1_dict = scio.loadmat(dataFile_1)
            data_1=dataFile_1_dict['train_data']
            print(data_1.shape)
            
            
            '''第二类数据'''
            dataFile_2 = file_path+r'\train2\train_data2.mat'
            dataFile_2_dict = scio.loadmat(dataFile_2)
            data_2=dataFile_2_dict['train_data2']
            print(data_2.shape)
            
            
            '''第三类数据'''
            dataFile_3 = file_path+r'\train2\train_data3.mat'
            dataFile_3_dict = scio.loadmat(dataFile_3)
            data_3=dataFile_3_dict['train_data3']
            print(data_3.shape)
            
            
            label1=np.array([0 for _ in range(52000)]).reshape(52000,1)
            data1=np.concatenate((data_1,label1),axis=1)
            
            label2=np.array([1 for _ in range(52000)]).reshape(52000,1)
            data2=np.concatenate((data_2,label2),axis=1)
            
            label3=np.array([2 for _ in range(36000)]).reshape(36000,1)
            data3=np.concatenate((data_3,label3),axis=1)
                
            train_data=np.concatenate((data1,data2,data3),axis=0)

            
            
            return_data = np.expand_dims(train_data[:,0:256],axis=1)

            
            return_data_label=train_data[:,512]
            self.data = return_data
            self.label=keras.utils.to_categorical(return_data_label, num_classes=3)
            
            
            dataFile_average = file_path+r'\train\train_average.mat'
            real_dict = scio.loadmat(dataFile_average)
            self.data_average=real_dict['train_average']  

            permutation = np.random.permutation(self.data.shape[0])
            self.data=self.data[permutation,:,:]
            self.label=self.label[permutation,:]
            self.data_average=self.data_average[permutation,:,:]
            
            print('data total shape',self.data.shape)
            '''
            permutation = np.random.permutation(self.data.shape[0])

            # shuffled_dataset = train_data[permutation, :, :]
            # shuffled_labels = train_label[permutation]
            
            '''
class train_dataloader_mult_prior():
    '''
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label_onehot = self.label_onehot[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]


            return_data_average = self.data_average[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label_onehot,return_data_average,return_label 
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label_onehot = self.label_onehot[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]

            return_data_average = self.data_average[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label_onehot,return_data_average,return_label     
        
    
    def load_train(self,file_path):
            '''第一类数据'''  
            dataFile_1 = file_path+r'\train2\train_data.mat'
            dataFile_1_dict = scio.loadmat(dataFile_1)
            data_1=dataFile_1_dict['train_data']
            print(data_1.shape)
            
            
            '''第二类数据'''
            dataFile_2 = file_path+r'\train2\train_data2.mat'
            dataFile_2_dict = scio.loadmat(dataFile_2)
            data_2=dataFile_2_dict['train_data2']
            print(data_2.shape)
            
            
            '''第三类数据'''
            dataFile_3 = file_path+r'\train2\train_data3.mat'
            dataFile_3_dict = scio.loadmat(dataFile_3)
            data_3=dataFile_3_dict['train_data3']
            print(data_3.shape)
            
            label1=np.array([0 for _ in range(52000)]).reshape(52000,1)
            data1=np.concatenate((data_1,label1),axis=1)
            
            label2=np.array([1 for _ in range(52000)]).reshape(52000,1)
            data2=np.concatenate((data_2,label2),axis=1)
            
            label3=np.array([2 for _ in range(36000)]).reshape(36000,1)
            data3=np.concatenate((data_3,label3),axis=1)
                
            train_data=np.concatenate((data1,data2,data3),axis=0)

            return_data = np.expand_dims(train_data[:,0:256],axis=1)

            
            return_data_label=train_data[:,256]
            self.label=return_data_label

            self.data = return_data
            self.label_onehot=keras.utils.to_categorical(return_data_label, num_classes=3)
                   
            
            
            dataFile_average = file_path+r'\train2\train_average.mat'
            real_dict = scio.loadmat(dataFile_average)
            self.data_average=real_dict['train_average']  
            self.data_average = np.expand_dims(self.data_average,axis=1)


            permutation = np.random.permutation(self.data.shape[0])
            self.data=self.data[permutation,:,:]
            self.label_onehot=self.label_onehot[permutation,:]
            self.label=self.label[permutation]
            self.data_average=self.data_average[permutation,:,:]
            
            print('data total shape',self.data.shape)
            '''
            permutation = np.random.permutation(self.data.shape[0])

            # shuffled_dataset = train_data[permutation, :, :]
            # shuffled_labels = train_label[permutation]
            
            '''            
 
            
class train_data_generator1():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第一类数据'''  
            dataFile_real1 = file_path+r'\train2\train_data.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['train_data']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([0 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
                     
   
class train_data_generator2():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第2类数据'''  
            dataFile_real1 = file_path+r'\train2\train_data2.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['train_data2']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([1 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
                     
                       
class train_data_generator3():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第3类数据'''  
            dataFile_real1 = file_path+r'\train2\train_data3.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['train_data3']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([3 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
                     
            
            
class train_data_generator_t3():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第3类数据'''  
            dataFile_real1 = file_path+r'\test2\test_data3.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['test_data3']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([2 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
                     
            
class train_data_generator_t2():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第2类数据'''  
            dataFile_real1 = file_path+r'\test2\test_data2.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['test_data2']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([1 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
 
 
class train_data_generator_t1():
    '''
    给自编码器数据，并把普通形式的label给自编码器
    '''
    def __init__(self,file_path,batch_size):
        self.batch_size=batch_size
        self.load_train(file_path)
        self.index=0
        self.length=len(self.data)
    
    def __next__(self):
        #print(self.index)
        if self.index < self.length-self.batch_size:
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
        else:
            self.index=0
            return_data = self.data[self.index:self.index+self.batch_size]
            return_label = self.label[self.index:self.index+self.batch_size]
            self.index+=self.batch_size
            return return_data,return_label
    
    def load_train(self,file_path):
            '''第一类数据'''  
            dataFile_real1 = file_path+r'\test2\test_data.mat'
            real_dict = scio.loadmat(dataFile_real1)
            data_real1=real_dict['test_data']
            print(data_real1.shape)
            data1=data_real1
            
            label1=np.array([0 for _ in range(len(data_real1))]).reshape(len(data_real1),1)
            data1=np.concatenate((data_real1,label1),axis=1)
            
  
            
            test_data=data1            
            return_data_real = np.expand_dims(test_data[:,0:256],axis=1)
            return_data_label=test_data[:,256]
            self.data =return_data_real
            self.label=return_data_label
            
    
def load_test(file_path):
    '''第一类数据'''  
    dataFile_real1 = file_path+r'\test\test_data_real.mat'
    real_dict = scio.loadmat(dataFile_real1)
    data_real1=real_dict['test_data_real'].T
    
    dataFile_imag1 =file_path+ r'\test\test_data_imag.mat'
    imag_dict = scio.loadmat(dataFile_imag1)
    data_imag1=imag_dict['test_data_imag'].T
    
    '''第二类数据'''
    dataFile_real2 =file_path+ r'\test\test_data_real2.mat'
    real_dict = scio.loadmat(dataFile_real2)
    data_real2=real_dict['test_data_real2']
    
    dataFile_imag1 = file_path+ r'\test\test_data_imag2.mat'
    imag_dict = scio.loadmat(dataFile_imag1)
    data_imag2=imag_dict['test_data_imag2']
    
    '''第三类数据'''
    dataFile_real3 =file_path+ r'\test\test_data_real3.mat'
    real_dict = scio.loadmat(dataFile_real3)
    data_real3=real_dict['test_data_real3']
    
    dataFile_imag3 =file_path+ r'\test\test_data_imag3.mat'
    imag_dict = scio.loadmat(dataFile_imag3)
    data_imag3=imag_dict['test_data_imag3']
    
    label1=np.array([0 for _ in range(138240)]).reshape(138240,1)
    data1=np.concatenate((data_real1,data_imag1,label1),axis=1)
    
    label2=np.array([1 for _ in range(138200)]).reshape(138200,1)
    data2=np.concatenate((data_real2,data_imag2,label2),axis=1)
    
    label3=np.array([2 for _ in range(138240)]).reshape(138240,1)
    data3=np.concatenate((data_real3,data_imag3,label3),axis=1)
    
    test_data=np.concatenate((data1,data2,data3),axis=0)
    np.random.shuffle(test_data)
    
    return_data_real = np.expand_dims(test_data[0:100,0:256],axis=2)
    return_data_imag = np.expand_dims(test_data[0:100,256:512],axis=2)
    return_data_label=test_data[0:100,512]
    X=np.concatenate((return_data_real,return_data_imag),axis=2)
    Y=keras.utils.to_categorical(return_data_label, num_classes=3)
    return X,Y