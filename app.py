#!/usr/bin/env python
# coding: utf-8


import streamlit as st 
import pandas as pd
import numpy as np


st.title('Customer Churn Prediction')
st.markdown('Toy model to play to predict churn of Customers')


st.header("Features")
col1, col2 = st.columns(2)

with col1:
    st.text("Characteristics")
    C_1 = st.slider('Subcription lenght (Months)', 1, 12 , 24)
    C_2 = st.slider('Monthly Bill (Unit)', 30, 70, 100 )
 
with col2:
   st.text("Oher Characteristics")
   O_2 = st.slider('Usage(GB)', 50, 250, 500)


st.text('')
if st.button("Predict Churn"):
    result = predict(
        np.array([[C_1, C_2, O_2]]))
    st.text(result[0])






