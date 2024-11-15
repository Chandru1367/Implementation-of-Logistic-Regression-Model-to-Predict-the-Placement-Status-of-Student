# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.  Import the required packages and print the present data.
 2.  Print the placement data and salary data.
 3.  Find the null and duplicate values.
 4.  Using logistic regression find the predicted values of accuracy , confusion matrices.
 5.  Display the results

## Program:
```python
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Chandru
RegisterNumber: 24900224
```
```python
 import pandas as pd
 data=pd.read_csv("Placement_Data (1).csv")
 print(data.head())
 data1=data.copy()
 data1=data1.drop(["sl_no","salary"],axis=1)
 print(data1.head())
 data1.isnull().sum()
 data1.duplicated().sum()
 from sklearn.preprocessing import LabelEncoder
 le=LabelEncoder()
 data1["gender"]=le.fit_transform(data1["gender"])
 data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
 data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
 data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
 data1["degree_t"]=le.fit_transform(data1["degree_t"])
 data1["workex"]=le.fit_transform(data1["workex"])
 data1["specialisation"]=le.fit_transform(data1["specialisation"])
 data1["status"]=le.fit_transform(data1["status"])
 print(data1)
 x=data1.iloc[:,:-1]
 x
 y=data1["status"]
 y
 from sklearn.model_selection import train_test_split
 x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
 from sklearn.linear_model import LogisticRegression
 lr=LogisticRegression(solver="liblinear")
 lr.fit(x_train,y_train)
 y_pred=lr.predict(x_test)
 print(y_pred)
 from sklearn.metrics import accuracy_score
 accuracy=accuracy_score(y_test,y_pred)
 print(accuracy)
 from sklearn.metrics import confusion_matrix
 confusion=confusion_matrix(y_test,y_pred)
 print(confusion)
 from sklearn.metrics import classification_report
 classification_report1=classification_report(y_test,y_pred)
 print(classification_report1)
 lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
 HEAD
 
![Screenshot 2024-11-15 100329](https://github.com/user-attachments/assets/a7031dc5-5360-49d3-82ac-8a195a59887a)

COPY

![Screenshot 2024-11-15 100338](https://github.com/user-attachments/assets/0e7c4e28-03c0-4f79-bc11-7e8adc0959a8)

FIT TRANSFORM

![Screenshot 2024-11-15 100430](https://github.com/user-attachments/assets/596dc3e3-1efd-4ea4-9e8e-3537d1b7222e)

LOGISTIC REGRESSION

![Screenshot 2024-11-15 181611](https://github.com/user-attachments/assets/be783b18-eca1-4fda-8f92-8ba5ce714673)

ACCURACY SCORE

![Screenshot 2024-11-15 183122](https://github.com/user-attachments/assets/a982bc3f-973a-4565-8ba2-7b4b2fb26d5d)

CONFUSION MATRIX

![Screenshot 2024-11-15 181630](https://github.com/user-attachments/assets/4a21c535-feea-4cb4-bd91-04ed12ee521b)

CLASSIFICATION REPORT

![Screenshot 2024-11-15 183250](https://github.com/user-attachments/assets/186f0135-0c0f-46d0-be46-dfca7798206c)

PREDICTION

![Screenshot 2024-11-15 183309](https://github.com/user-attachments/assets/e75282d7-766a-40c1-b531-fa3c1d695207)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
