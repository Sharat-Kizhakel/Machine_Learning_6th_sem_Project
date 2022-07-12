import machine_learning_2nd_week as ml2
import streamlit as st
import pandas as pd
import csv
import base64
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from PIL import Image

st.title("Breast Cancer Detection results:")
st.write("")
st.write("")

# st.write("""
# # Explore different classifier
# Which one is the best?
# """)
# dataset_name = st.sidebar.button("Upload dataset")
# if st.sidebar.button("Upload dataset"):
dataset = st.sidebar.file_uploader(
    label="Upload your dataset", type=["csv", "txt"])


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background('breast_cancer.png')

if dataset is not None:
    dataset_df = pd.read_csv(dataset)
    # print(dataset_df)
    st.write(dataset_df)
else:
    # image = Image.open('breast_cancer.png')
    # st.image(image, caption='Classify by uploading dataset!')
    pass
    # st.write(dataset_name)
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
st.sidebar.write("")
classifier_name = st.sidebar.selectbox(
    "Select Classifier", ("Decision Tree", "Random Forest", "Logistic Regression"))
st.sidebar.write("")
st.sidebar.write("")
visualization_name = st.sidebar.selectbox(
    "Select Visualization", ("Count Plot", "Heat Map", "Pair Plot"))
if st.sidebar.selectbox("Select Prediction:", ("Predict", "Predict")):
    st.title('Breast Cancer Prediction using ML')
    with st.form(key='form1'):
        # getting the input data from the user
        print("Inside form")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            mean_radius = st.text_input('mean radius')
            print(mean_radius)

        with col1:
            mean_texture = st.text_input('mean texture')

        with col1:
            mean_perimeter = st.text_input('mean perimeter')

        with col1:
            mean_area = st.text_input('mean area')

        with col1:
            mean_smoothness = st.text_input('mean smoothness')

        with col1:
            mean_compactness = st.text_input('mean compactness')

        with col2:
            mean_concavity = st.text_input('mean concavity')
        with col2:
            mean_concave_points = st.text_input('mean concave points')

        with col2:
            mean_symmetry = st.text_input('mean symmetry')
        with col2:
            mean_fractal_dimension = st.text_input('mean fractal dimension')
        with col2:
            radius_error = st.text_input('radius error')
        with col2:
            texture_error = st.text_input('texture error')

        with col3:
            perimeter_error = st.text_input('perimeter error')
        with col3:
            area_error = st.text_input('area error')
        with col3:
            smoothness_error = st.text_input('smoothness error')
        with col3:
            compactness_error = st.text_input('compactness error')
        with col3:
            concavity_error = st.text_input('concavity error')
        with col3:
            concave_points_error = st.text_input('concave points error')

        with col4:
            symmetry_error = st.text_input('symmetry error')

        with col4:
            fractal_dimension_error = st.text_input('fractal dimension error')

        with col4:
            worst_radius = st.text_input('worst radius')

        with col4:
            worst_texture = st.text_input('worst texture')

        with col4:
            worst_perimeter = st.text_input('worst perimeter')

        with col4:
            worst_area = st.text_input('worst area')
            print(worst_area)
        with col5:
            worst_smoothness = st.text_input('worst smoothness')
            print(worst_smoothness)
        with col5:
            worst_compactness = st.text_input('worst compactness')
            print(worst_compactness)
        with col5:
            worst_concavity = st.text_input('worst concavity')
            print(worst_concavity)
        with col5:
            worst_concave_points = st.text_input('worst concave points')
            print(worst_concave_points)
        with col5:
            worst_symmetry = st.text_input('worst symmetry')
        with col5:
            worst_fractal_dimension = st.text_input('worst fractal dimension')
            print(worst_fractal_dimension)

        # creating a button for Prediction
        TEST = [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error,
                concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry]
        #  extra parameter worst_fractal_dimension
        print(len(TEST))
        print(worst_fractal_dimension)

        submit_button = st.form_submit_button(label='Breast Cancer Diagnosis')

        if submit_button:

            print(submit_button)
            st.write("Checking for chances of cancer:")
            pred = ml2.model[1].predict([[mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error,
                                        compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry]])
            # [mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry].reshape())
            print(pred)
            if pred == 1:
                cancer_diagnosis = 'The patient has breast cancer'
                st.success(cancer_diagnosis)
            else:
                cancer_diagnosis = 'The patient does not have breast cancer'
                st.success(cancer_diagnosis)
# else:
#     pass

if dataset is not None:
    def get_dataset(dataset_name):
        # if dataset_name == "data.csv":
        print("inside get dataset", dataset)
        df = pd.read_csv(dataset_name)  # cant access relative path files
        X = df.iloc[:, 2:31].values
        y = df.iloc[:, 1].values
        return X, y

    print("In get function", dataset.name)
    X, y = get_dataset(dataset.name)
    st.write("Shape of data:", X.shape)
    st.write("")
    st.write("Number of Classes:", len(np.unique(y)))
    st.write("")

# def display_parameter(classifier_name):

#     parameters = dict()
#     if classifier_name == "KNN":
#         K = st.sidebar.slider("K", 1, 15)
#         parameters["K"] = K

    def visualization_display(visualization_name):

        if visualization_name == "Count Plot":

            st.title(visualization_name)
            fig = plt.figure(figsize=(10, 4))
            sns.countplot(ml2.df['diagnosis'], label="count")

            st.pyplot(fig)
        elif visualization_name == "Heat Map":
            st.title(visualization_name)
            fig, ax = plt.subplots()
            sns.heatmap(ml2.df.iloc[:, 1:10].corr(), annot=True, fmt=".0%")
            st.write(fig)

        elif visualization_name == "Pair Plot":

            st.title(visualization_name)
            fig = sns.pairplot(ml2.df.iloc[:, 1:5], hue="diagnosis")
            st.pyplot(fig)

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
    visualization_display(visualization_name)
else:
    # st.warning("You need to upload a csv file first")
    pass
# ml2.ml2.models(ml2.X_train, ml2.Y_train)
