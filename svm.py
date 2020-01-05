import pandas as pd
dataset=pd.read_csv(r'F:\mukul ml\aNN\speech\speech.csv')
test=pd.read_csv(r'F:\mukul ml\aNN\speech\test.csv')

xt=test.iloc[0:1,0:13].values
x=dataset.iloc[:,1:13].values
y=dataset.iloc[:,13].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)
xt=sc.transform(xt)
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
newt=classifier.predict(xt)
print(newt)