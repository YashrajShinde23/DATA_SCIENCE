import pandas as pd
from sklearn.preprocessing import StandardScaler
from numpy.linalg import svd
import matplotlib.pyplot as plt

#step 1: load the dataset
df=pd.read_csv("heart disease.csv")

#step 2: separate features and target
X=df.drop(columns=["target"])#all features
y=df["target"] #target 0&1

#step 3: standardize the freature matrix
scaler= StandardScaler()
X_scaled=scaler.fit_transform(X)

#step 4: apply singular value decomposition
U,S,VT=svd(X_scaled,full_matrices=False)

#compute variance explained by each component
explained_variance=(S**2)/(X_scaled.shape[0]-1)
total_variance=explained_variance.sum()
explained_variance_ratio=explained_variance/total_variance

#display variance explained by each component
for i,ev in enumerate(explained_variance_ratio):
    print(f"component {i+1}: {ev:4f} ({ev*100:.2f}%)")


#step 5: project data onto first 2 SVD components
X_svd_2d=U[:,:2]*S[:2] #2d projection for plotting

#step 6: plotting
plt.figure(figsize=(8,6))
for label in sorted(y.unique()):
    plt.scatter(X_svd_2d[y==label,0],X_svd_2d[y==label,1],label=f"class{label}",alpha=0.6)
    
plt.title("SVD: projection onto first 2 componets (heart dataset)")
plt.xlabel("component 1")
plt.ylabel("component 2")
plt.legend()
plt.grid(True)
plt.show()