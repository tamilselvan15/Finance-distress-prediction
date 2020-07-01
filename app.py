# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 01:17:58 2020

@author: tamil
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st 

from lightgbm import LGBMClassifier

pickle_in = open("lgbm.sav","rb")
hybrid_model = pickle.load(pickle_in)

def main(): 
    
    st.title("Finance Distress Predictor") 
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Finance Distress Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    debt_limit_ratio = st.text_input("Revolving Utilization Of UnsecuredLine","Type Here")
    age = st.text_input("Age","Type Here")
    income = st.text_input("Monthly Income" ,"Type Here")
    debt = st.text_input("Debt ratio in %","Type Here")
    tf = st.text_input("Number Of Time 30-59 Days Past Due","Type Here")
    dep = st.text_input("Number Of Dependents","Type Here")
    
    any_due = ["Yes","No"]
    due = st.selectbox("Any Past Due?",any_due)
    
    
    if due == "Yes":
        past_due = 1
        
    if due == "No":
        past_due = 0

        
        
    loans = ["Yes","No"]
    get_loan = st.selectbox("No. of Loans More Than 5?",loans)
    
    
    if get_loan == "Yes":
        com_loan = 1
        
    if get_loan == "No":
        com_loan = 0
    
    if st.button("Predict"):
        prediction=hybrid_model.predict_proba([[float(debt_limit_ratio),int(age),
                                   int(income),float(debt),
                                   int(tf),int(dep),
                                   past_due,com_loan,]])
        
        p = prediction[:,1]
        if p > 0.5:
            st.warning('{}% Chance of risk'.format((p)*[100]))
        else:
            st.success('{}% Chance of risk'.format((p)*[100]))
    
    
if __name__=='__main__':
    main()
