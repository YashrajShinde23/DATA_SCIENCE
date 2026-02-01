import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import warnings
warnings.filterwarnings("ignore")
from sklearn import datasets
iris=datasets.load_iris()
X_train,y_train=iris.data[:,1:3],iris.target #taking entire data as training
clf1=LogisticRegression()
clf2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()
##############################
print("After five fold vross validation")
labels=["Logistics Regresion","Random Forest model","Naive Bayes model"]
#the labels list is simply a list of names corresponding to the

for clf,label in zip([clf1,clf2,clf3],labels):
    scores=model_seletion.cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    print("Accuracy",scores.mean(),"for ",label)
    
voting_clf_hard=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                             voting="hard"
                                             
                                             )
voting_clf_soft=VotingClassifier(estimators=[(labels[0],clf1),
                                             (labels[1],clf2),
                                             (labels[2],clf3)],
                                             voting="soft"
                                             
                                             )
labels_new=["Logistic Regression","Random Forest model","Naive Bayes model","voting_clf_hard","voting_clf_soft"]

#names for all five models(3 base + 2 ensemble models) 
for clf,label in zip([clf1,clf2,clf3,voting_clf_hard,voting_clf_soft],labels_new):
    scores=model_selection.cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    print("Accuracy: ",scores.mean(),"for ",label)                                            
                