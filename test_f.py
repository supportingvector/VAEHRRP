#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 10 20:48:03 2018

@author: xms
"""
import sys
import numpy as np
sys.path.append(r'C:\Program Files\Anaconda3\Lib\site-packages\libsvm\python')
from libsvm.python.svmutil import *
from libsvm.python.svm import *  
#y, x = np.array([1,-1]), np.array([[1,2,2], [1,2,3]])
#prob  = svm_problem(y, x)
##param = svm_parameter('-t 0 -c 4 -b 1')
#print("((((((((((((((((")
#print(prob)
#model = svm_train(y,x)
#yt = [1]
#xt = [{1:1, 2:1}]
#p_label, p_acc, p_val = svm_predict(yt, xt, model)
#print(p_label)
def test_accuracy():
    z1=np.load('save\\z1.npy').tolist()
    z2=np.load('save\\z2.npy').tolist()
    z3=np.load('save\\z3.npy').tolist()
    x=z1+z2+z3
    #x=z1+z2
    y1=[0 for i in range(52000)]
    y2=[1 for i in range(52000)]
    y3=[2 for i in range(36000)]    
    y=y1+y2+y3
    #y=y1+y2
    model = svm_train(y,x)
    
    z1test=np.load('save\\z1t.npy').tolist()
    z2test=np.load('save\\z2t.npy').tolist()
    z3test=np.load('save\\z3t.npy').tolist()
    xtest=z1test+z2test+z3test
    #xtest=z1test+z2test
    yt1=[0 for i in range(2000)]
    yt2=[1 for i in range(2000)]
    yt3=[2 for i in range(1200)]    
    ytest=yt1+yt2+yt3
    #ytest=yt1+yt2
    #p_label, p_acc, p_val = svm_predict(y, x, model)
    p_label, p_acc, p_val = svm_predict(ytest, xtest, model)
    c1_1=p_label[0:2000].count(0.0)/2000
    c1_2=p_label[0:2000].count(1.0)/2000
    c1_3=p_label[0:2000].count(2.0)/2000
    
    c2_1=p_label[2000:4000].count(0.0)/2000
    c2_2=p_label[2000:4000].count(1.0)/2000
    c2_3=p_label[2000:4000].count(2.0)/2000
    
    c3_1=p_label[4000:5200].count(0.0)/2000
    c3_2=p_label[4000:5200].count(1.0)/2000
    c3_3=p_label[4000:5200].count(2.0)/1200
    
    acc=(c1_1+c2_2+c3_3)/3


#p_label, p_acc, p_val = svm_predict(y, x, model)
#print(p_label)







    
    
      