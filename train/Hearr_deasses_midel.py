import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib


data  = pd.read_csv("heart.csv")

X = data.drop("target",axis=1)
y = data["target"]


# Split the data for test and train
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# train  the model
model = LogisticRegression(max_iter=100)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)


# check the effienct
accuracy = accuracy_score(y_test,y_pred)
print(accuracy)
print(classification_report(y_test,y_pred))


# save the model
joblib.dump(model,"heart_desease.pkl")
