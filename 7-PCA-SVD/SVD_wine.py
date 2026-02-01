
import pandas as pd
#load the dataset
df=pd.read_csv("c:/Data-Science/7-PCA-SVD/wine.csv")
print(df.head())
#drop the target column (assumed to be 'Type')
x=df.drop(columns=['Type'])#feature
y=df['Type']
#step3 Standardize the data

from sklearn.preprocessing import StandardScaler


#perform svd
from numpy.linalg import svd
#step 4 preform SVD
U,S,VT=svd(x_scaled,full_matrices==False)
#U=left singular vector (sample*component)
#s:singluar value
#VT: right singluar  vector
#step 5: print shapes of decomposed mattrices
#step6 explained variance calcluation
import numpy as np
x_scaled.shape
#compute variance explained by each component
explained_variance=explained_variance.sum()
total_variance=explained_variance.sum()
explained_variance_ratio=explained_variance/total_variance


#display variance explained by each compont
for i,ev in  enumerate(explained_variance_ratio):
    print(f"Componet {i+1}:{ev:4f}({ev*100:2f}%)")
#step7 project data onto 1st 2 component (for visualization)
#1st 2 component projection
x_svd_2d=U[:,:2]*S[:2] #reconstruct 2D projection

import matplotlib.pyplot as plt
plt.figurefigsize=(8,6))
for lable in sorted(y.unique()):
    plt.scatter(x_svd_2d[y==label,0],x_svd_2d[y==lable,1],lable=f






