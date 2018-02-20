# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 16:56:23 2018

@author: MONIK RAJ
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split

#Loading the dataset
data = pd.read_csv('CASP.csv')

corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
corr_plot_matrix = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

# save the figure
fig = corr_plot_matrix.get_figure()
fig.savefig('ME-CORR.png')

# formation of X and Y from data frame to numpy array
Y = data['RMSD']
del data['RMSD']
X = data

Y = np.array(Y).reshape(len(Y),1)
X = np.array(X).reshape(len(X),9)

k=[]
vs = []
prevset = []
#The list containing features in descending order of their preference in selectKBest()
kfeatures = []
#X_best = SelectKBest(f_regression, k=5).fit_transform(X,Y)

'''
# Code snippet to get r2 score of the model for various values of K
for i in range(0,9):
    selector = SelectKBest(f_regression, k=i+1)
    X_best = selector.fit_transform(X,Y)
    newset = selector.get_support(indices=True) 
    kfeatures.append(list(set(newset)-set(prevset))[0]+1)
    prevset=newset
    X_train, X_test, Y_train, Y_test = train_test_split(X_best, Y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.reshape(len(Y_pred),1)
    k.append(i+1)
    vs.append(r2_score(Y_test, Y_pred))
'''
#function to find covariance matrix    
def covariance(X):
    d = X.shape[1]
    cov = np.zeros((d,d))
    def covar(U,V):
        Umean = np.mean(U)
        Vmean = np.mean(V)
        s = np.dot((U-Umean),(V-Vmean).T)
        s = s / len(U)
        return s
    for i in range(0,d):
        for j in range(0,d):
            cov[i][j] = covar(X[:,i], X[:,j])
    return cov 

sigma = covariance(X)
eigval, eigvector = np.linalg.eig(sigma)
idx = eigval.argsort()[::-1]   
eigvalSort = eigval[idx]
eigvectorSort = eigvector[:,idx]

#For selecting only two principal components for visualization
X_PCA_2 = np.dot(eigvectorSort[:,0:2].T,X.T).T
                 
X1, X2 = np.meshgrid(X_PCA_2[0:45730:100,0:1].reshape(458,),X_PCA_2[0:45730:100,1:2].reshape(458,)) 
# Make the plot 3D
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X1, X2, Y[0:45730:100].reshape(458,), cmap=cm.coolwarm,linewidth=0, antialiased=False)

fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
fig.savefig("ME-PCA_02.png")


m=[]
r2L=[]

for i in range(1,51):
    data = pd.read_csv('CASP.csv')
    f = float(float(2*i)/float(100))
    data_m = data.sample(frac=f,replace=False)
    Y = data_m['RMSD']
    del data_m['RMSD']
    X = data_m    
    Y = np.array(Y).reshape(len(Y),1)
    X = np.array(X).reshape(len(X),9)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.reshape(len(Y_pred),1)
    k.append(f)
    r2L.append(r2_score(Y_test, Y_pred))
