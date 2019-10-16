#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

z1=np.load('z1.npy')
pca = PCA(n_components=2)
pcaz1=pca.fit_transform(z1)
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

    
z2=np.load('z2.npy')
pca = PCA(n_components=2)
pcaz2=pca.fit_transform(z2)
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))

z3=np.load('z3.npy')
pca = PCA(n_components=2)
pcaz3=pca.fit_transform(z3)
# PCA(copy=True, n_components=2, whiten=False)
print(pca.explained_variance_ratio_)
print(sum(pca.explained_variance_ratio_))    

plt.scatter(pcaz1[:,0], pcaz1[:,1], s=10,marker='o',c='',edgecolors='y')
plt.scatter(pcaz2[:,0], pcaz2[:,1], s=20,marker='o',c='',edgecolors='r')
plt.scatter(pcaz3[:,0], pcaz3[:,1], s=20,c='b',marker='x')
#     
plt.show() 
    
#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#ax.scatter(pcaz1[:,0], pcaz1[:,1],pcaz1[:,2],c='b')
#ax.scatter(pcaz1[:,0], pcaz1[:,1],pcaz1[:,2],c='r')
#ax.scatter(pcaz1[:,0], pcaz1[:,1],pcaz1[:,2],c='k')
#plt.show()

