import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

st.write("""
# Iris Flower Prediction Using Different Classifiers

This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

iris = datasets.load_iris()
X = iris.data
Y = iris.target

st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)

def predicting(clf):
    clf.fit(X,Y)
    prediction = clf.predict(df)
    st.write('''**Prediction: **'''+ iris.target_names[prediction][0])

def DTC():
    clf = DecisionTreeClassifier()
    predicting(clf)

def GNBC():
    clf = GaussianNB()
    predicting(clf)

def RFC():
    clf = RandomForestClassifier()
    predicting(clf)

def KNC():
    clf = KNeighborsClassifier(n_neighbors = 3)
    predicting(clf)

def SVMC():
    clf = SVC(kernel = 'linear')
    predicting(clf)

def LR():
    clf = LogisticRegression(solver = 'newton-cg')
    predicting(clf)

if st.button('Use Decision Tree Classifier'):
    DTC()
    

if st.button('Use Gaussian Naive Bayes Classifier'):
    GNBC()
    
    
if st.button('Use Random Forest Classifier'):
    RFC()
    
if st.button('Use K-Neighbors Classifier'):
    KNC()

if st.button('Use Linear SVC'):
    SVMC()
    
if st.button('Use Logistic Regression'):
    LR()
    
if st.button('Summary'):
    classifiers = ['Decision Tree','Naive Bayes', 'Random Forest', 'K-Neighbors','Linear SVC', 'Logistic Regression']
    predicted_values = []
    clf = DecisionTreeClassifier()
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])

    clf = GaussianNB()
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])
    
    clf = RandomForestClassifier()
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])

    clf = KNeighborsClassifier(n_neighbors = 3)
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])
    
    clf = SVC(kernel = 'linear')
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])

    clf = LogisticRegression(solver = 'newton-cg')
    clf.fit(X,Y)
    predicted_values = np.concatenate([predicted_values,iris.target_names[clf.predict(df)]])
    
    predicted_values_df = pd.DataFrame(predicted_values, columns= ['Prediction'],index = classifiers)
    st.dataframe(predicted_values_df)
