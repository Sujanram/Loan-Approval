import pandas as pd 
import numpy as np                      
import seaborn as sns                 
import matplotlib.pyplot as plt 


train=pd.read_csv("../train_ctrUa4K.csv") 
test=pd.read_csv("../test_lAUu6dG.csv")

train_original=train.copy() 
test_original=test.copy()

def name(arg):
    train[arg].fillna(train[arg].mode()[0], inplace=True)
name('Gender')
name('Married')
name('Dependents')
name('Self_Employed')
name('Credit_History')
name('Loan_Amount_Term')
train['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)

def name1(arg):
    test[arg].fillna(train[arg].mode()[0], inplace=True)
name1('Gender')
name1('Married')
name1('Dependents')
name1('Self_Employed')
name1('Credit_History')
name1('Loan_Amount_Term')
test['LoanAmount'].fillna(train['LoanAmount'].median(), inplace=True)


train=train.drop('Loan_ID',axis=1) 
test=test.drop('Loan_ID',axis=1)



X = train.drop('Loan_Status',1) 
y = train.Loan_Status

X1=pd.get_dummies(X['Gender'],prefix='Gender',drop_first=True) 
X=X.drop('Gender',axis=1)
X2=pd.get_dummies(X['Married'],prefix='Married',drop_first=True) 
X=X.drop('Married',axis=1)
X3=pd.get_dummies(X['Education'],prefix='Education',drop_first=True) 
X=X.drop('Education',axis=1)
X4=pd.get_dummies(X['Self_Employed'],prefix='Self_Employed',drop_first=True) 
X=X.drop('Self_Employed',axis=1)

X=pd.concat([X1,X],axis=1)
X=pd.concat([X3,X],axis=1)
X=pd.concat([X4,X],axis=1)
X=pd.concat([X2,X],axis=1)

X1=pd.get_dummies(train['Gender'],prefix='Gender',drop_first=True) 
train=train.drop('Gender',axis=1)
X2=pd.get_dummies(train['Married'],prefix='Married',drop_first=True) 
train=train.drop('Married',axis=1)
X3=pd.get_dummies(train['Education'],prefix='Education',drop_first=True) 
train=train.drop('Education',axis=1)
X4=pd.get_dummies(train['Self_Employed'],prefix='Self_Employed',drop_first=True) 
train=train.drop('Self_Employed',axis=1)

train=pd.concat([X1,train],axis=1)
train=pd.concat([X3,train],axis=1)
train=pd.concat([X4,train],axis=1)
train=pd.concat([X2,train],axis=1)

X1=pd.get_dummies(test['Gender'],prefix='Gender',drop_first=True) 
test=test.drop('Gender',axis=1)
X2=pd.get_dummies(test['Married'],prefix='Married',drop_first=True) 
test=test.drop('Married',axis=1)
X3=pd.get_dummies(test['Education'],prefix='Education',drop_first=True) 
test=test.drop('Education',axis=1)
X4=pd.get_dummies(test['Self_Employed'],prefix='Self_Employed',drop_first=True) 
test=test.drop('Self_Employed',axis=1)

test=pd.concat([X1,test],axis=1)
test=pd.concat([X3,test],axis=1)
test=pd.concat([X4,test],axis=1)
test=pd.concat([X2,test],axis=1)

X['Dependents']=X['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
train['Dependents']=train['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})
test['Dependents']=test['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})

X['Property_Area']=X['Property_Area'].map({'Semiurban': '1', 'Urban': '2','Rural': '3'})
train['Property_Area']=train['Property_Area'].map({'Semiurban': '1', 'Urban': '2','Rural': '3'})
test['Property_Area']=test['Property_Area'].map({'Semiurban': '1', 'Urban': '2','Rural': '3'})



from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(X,y, test_size =0.3)

from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, max_iter=100,
                   multi_class='ovr', n_jobs=1,penalty='l2', random_state=1, solver='liblinear', tol=0.0001,
                   verbose=0, warm_start=False)
pred_cv = model.predict(x_cv)
accuracy_score(y_cv,pred_cv)
pred_test = model.predict(test)



import joblib
logit_model = open("logit_mod.pkl","wb")
joblib.dump(model,logit_model)
logit_model.close()
