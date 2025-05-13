# app.py
import streamlit as st
import joblib
import numpy as np

# Load mô hình
model = joblib.load("iris_model.pkl")

# Giao diện web
st.title("Dự đoán loài hoa Iris 🌸")

st.write("Nhập thông tin đặc trưng của hoa:")

sepal_length = st.slider("Chiều dài đài hoa (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("Chiều rộng đài hoa (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("Chiều dài cánh hoa (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("Chiều rộng cánh hoa (cm)", 0.1, 2.5, 1.2)

if st.button("Dự đoán"):
    X_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(X_input)[0]
    classes = ['Setosa', 'Versicolor', 'Virginica']
    st.success(f"Loài hoa dự đoán: **{classes[prediction]}**")
