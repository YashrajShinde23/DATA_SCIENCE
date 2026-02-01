import pandas as pd
df=pd.read_csv("movies_classifier.csv")
df.head()
df.columns
df.dtypes
#there are two columns of object type
df=pd.get_dummies(df,columns=["3D_available","Genre"],drop_first=True)
#########
#assign input and output
predictors=df.loc[:,df.columns!="Start_Tech_Oscar"]
target=df["Start_Tech_Oscar"]
###################
#train_test_split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(predictors,target,test_size=0.2,random_state=0)
from sklearn.ensemble import GradientBoostingClassifier
grand_boost=GradientBoostingClassifier()

grand_boost.fit(X_train,y_train)
#evaluation of model
pred1=grand_boost.predict(X_test)
from sklearn.metrics import accuracy_score,confusion_matrix
accuracy_score(y_test,pred1)
confusion_matrix(y_test,pred1)
######################
#evalution on training data
pred2=grand_boost.predict(X_train)
accuracy_score(y_train,pred2)
##################################
#let us change the hyper parameters
grand_boost1=GradientBoostingClassifier(learning_rate=0.02,n_estimators=5000,max_depth=1)
'''
learning_rate=0.02
controls how much each tree contribures to the overall prediction.
lower value -> slower learning,but potentially better accuracy.
works well with larger n_estimators.
n_estimators=5000
number of boosting rounds(or trees).
since learning_rate is small,more trees are needed to fit the data.
max_depth=1
each tree is a shallow tree(decision stump).
learns only simple rules (1 split per tree)
this help the model learn gradually and avoid overfitting.
'''

grand_boost1.fit(X_train,y_train)
'''
trains the gradientboostingclassifier on the features and target.
each tree is built sequentially to correct
the errors of the previous model.
after 5000 trees, you get the final boosted model.
'''
####################
from sklearn.metrics import accuracy_score,confusion_matrix
pred3=grand_boost1.predict(X_test)
accuracy_score(y_test,pred3)
confusion_matrix(y_test,pred3)
########################
#evalution on training data
pred4=grand_boost1.predict(X_train)
accuracy_score(y_train,pred4)