import streamlit as st
import pandas as pd
import joblib

model=joblib.load("decison.pkl")
scaler=joblib.load("scaler.pkl")
colums=joblib.load("columns.pkl")


st.title("Heart Strock Prediction ")
st.markdown("Provide the Following Details")
age=st.slider("Age",18,100,40)#("name",starting,ending,choosen)
sex=st.selectbox("Sex",['Male','Female'])
chol=st.number_input("Cholestrol(mg/dl)",100,600,200)
fbs=st.selectbox("Fasting Blood Sugar",[0,1])
oldpeck=st.slider("Oldpeak(ST depression)",0.0,0.6,1.0)


if st.button("Predict"):#'age', 'sex', 'chol', 'fbs', 'oldpeak', 'target'
    raw_input={
        "age":age,
        "sex":sex,
        "chol":chol,
        "fbs":fbs,
        "oldpeak":oldpeck
    }
    input_df=pd.DataFrame([raw_input])