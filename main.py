import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = pd.read_csv('Titanic-Dataset.csv')
data['Age'] = data['Age'].fillna(int(data['Age'].mean()))


le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex']) 
data.drop(columns = ['PassengerId','Name','SibSp','Parch','Ticket','Fare','Cabin','Embarked'], inplace = True)
y = list(data['Survived'])
data.drop(columns=['Survived'], inplace=True)


#scalingg
ss = StandardScaler()
data = ss.fit_transform(data)
data = pd.DataFrame(data, columns = ['Pclass','Sex','Age'])

x = []
for i in range(len(list(data['Age']))):
   xx = [(list(data['Age']))[i], (list(data['Pclass']))[i], (list(data['Sex']))[i]]
   x.append(xx)
#splitting data
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=41)

model = DecisionTreeClassifier()
model.fit(xtrain, ytrain)
tested = model.predict(xtest) ####
count= 0 
for i in range(len(ytest)):
   if tested[i]==ytest[i]:
     count +=1
print(f'{count}/{len(ytest)}')

print(f"Accuracy {accuracy_score(ytest, tested)*100:.2f}")   
