This program aims to predict diabetes using a machine learning algorithm called K-Nearest Neighbors (KNN) Classifier. It takes in a dataset of 768 records containing information on patients' glucose level, insulin level, age, and other factors that could contribute to diabetes, and creates a KNN model to predict diabetes.

Installation
To run this program, you will need to have Python 3 installed on your computer, along with the following libraries:

  1. pandas
  2. matplotlib
  3. numpy
  4. sklearn

Usage
To use this program, follow these steps:

  1. Download the diabetes.csv dataset and save it in the same directory as the program.
  2. Run the program using a Python interpreter.
  3. The program will output the raw data before and after normalization, and then split the data into training and testing sets.
  4. The program will then create a KNN model with k=3 and test its accuracy.
  5. Finally, the program will test the accuracy of the KNN model for different values of k (from 1 to 10), and predict whether a new patient is likely to      have diabetes or not based on the input features.
