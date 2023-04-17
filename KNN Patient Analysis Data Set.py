import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("diabetes.csv")
data.head()



#determine y-axis
y = data.Outcome.values
x_rawData = data.drop(["Outcome"], axis=1)

#determine x-axis
x = (x_rawData - np.min(x_rawData)) / (np.max(x_rawData) - np.min(x_rawData))

print("Raw Datas before normalization\n")
print(x_rawData.head())

print("\n\n")

print("Raw Datas after normalization\n")
print(x.head())



#We will separate train data and test data
#We will use train data for differentiating healthy people and sick people
#We will use test data whether our machine learning model differentiate healthy people and sick people

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)

#creating KNN Model
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("K=3 is for, verifying our test data", knn.score(x_test, y_test))

#determine which k value is best for the test
counter = 1
for k in range(1,11):
    knn_new = KNeighborsClassifier(n_neighbors = k)
    knn_new.fit(x_train, y_train)
    print(counter, " ", "Accuracy Rate: ", knn_new.score(x_test, y_test)*100, "%")
    counter+=1



#Predict new patient
from sklearn.preprocessing import MinMaxScaler

#making normalization
sc = MinMaxScaler()
sc.fit_transform(x_rawData)

new_prediction = knn.predict(sc.transform(np.array([[6, 148, 72, 35, 0, 33.6, 0.627, 50]])))

new_prediction[0]





