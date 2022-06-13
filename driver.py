import machine_learning_2nd_week as ml2
import streamlit as st
import pandas as pd
import csv
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

st.title("Breast Cancer Detection results:")
st.write("")
st.write("")
# st.write("""
# # Explore different classifier
# Which one is the best?
# """)
dataset_name = st.sidebar.button("Wisconsin Dataset")
# st.write(dataset_name)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Decision Tree", "Random Forest", "Logistic Regression"))


def get_dataset(dataset_name):
    if dataset_name == "data.csv":
        df = pd.read_csv('data.csv')
        X = df.iloc[:, 2:31].values
        y = df.iloc[:, 1].values
    return X, y


X, y = get_dataset("data.csv")
st.write("Shape of data:", X.shape)
st.write("")
st.write("Number of Classes:", len(np.unique(y)))
st.write("")

# def display_parameter(classifier_name):

#     parameters = dict()
#     if classifier_name == "KNN":
#         K = st.sidebar.slider("K", 1, 15)
#         parameters["K"] = K


def accuracy_display(classifier_name):

    if classifier_name == "Logistic Regression":

        st.write("Model:", classifier_name)
        st.write(" ")
        st.write("Accuracy:", ml2.accuracy_score(
            ml2.Y_test, ml2.model[2].predict(ml2.X_test)))
    elif classifier_name == "Decision Tree":
        st.write("Model:", classifier_name)
        st.write(" ")
        st.write("Accuracy:", ml2.accuracy_score(
            ml2.Y_test, ml2.model[0].predict(ml2.X_test)))

    elif classifier_name == "Random Forest":
        st.write("Model:", classifier_name)
        st.write(" ")
        st.write("Accuracy:", ml2.accuracy_score(
            ml2.Y_test, ml2.model[1].predict(ml2.X_test)))


accuracy_display(classifier_name)
# ml2.ml2.models(ml2.X_train, ml2.Y_train)
