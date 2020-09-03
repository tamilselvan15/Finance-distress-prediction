# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 01:17:58 2020

@author: tamil
"""


import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import matplotlib.pyplot as plt
import seaborn as sns
import pybase64
plt.style.use("fivethirtyeight")

from lightgbm import LGBMClassifier

pickle_in = open("lgbm.sav","rb")
hybrid_model = pickle.load(pickle_in)


def Preprocessing(df):
    age_working = df.loc[(df["age"] >= 18) & (df["age"] < 60)]
    age_senior = df.loc[(df["age"] >= 60)]

    age_working_impute = age_working.MonthlyIncome.mean()
    age_senior_impute = age_senior.MonthlyIncome.mean()

    df["MonthlyIncome"] = np.absolute(df["MonthlyIncome"])
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(-1)

    df.loc[((df["age"] >= 18) & (df["age"] < 60)) & (df["MonthlyIncome"] == -1),\
                   "MonthlyIncome"] = age_working_impute
    df.loc[(df["age"] >= 60) & (df["MonthlyIncome"] == -1), "MonthlyIncome"] = age_senior_impute

    df["NumberOfDependents"] = np.absolute(df["NumberOfDependents"])
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
    df["NumberOfDependents"] = df["NumberOfDependents"].astype('int64')

    df["CombinedDefaulted"] = (df["NumberOfTimes90DaysLate"] + df["NumberOfTime60-89DaysPastDueNotWorse"])\
                                            + df["NumberOfTime30-59DaysPastDueNotWorse"]

    df.loc[(df["CombinedDefaulted"] >= 1), "CombinedDefaulted"] = 1

    df["CombinedCreditLoans"] = df["NumberOfOpenCreditLinesAndLoans"] + \
                                            df["NumberRealEstateLoansOrLines"]

    df.loc[(df["CombinedCreditLoans"] <= 5), "CombinedCreditLoans"] = 0
    df.loc[(df["CombinedCreditLoans"] > 5), "CombinedCreditLoans"] = 1

    df["WithDependents"] = df["NumberOfDependents"]
    df.loc[(df["WithDependents"] >= 1), "WithDependents"] = 1

    df["MonthlyDebtPayments"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["MonthlyDebtPayments"] = np.absolute(df["MonthlyDebtPayments"])
    df["MonthlyDebtPayments"] = df["MonthlyDebtPayments"].astype('int64')

    df["age_map"] = df["age"]
    df.loc[(df["age"] >= 18) & (df["age"] < 60), "age_map"] = 1
    df.loc[(df["age"] >= 60), "age_map"] = 0

    #replacing those numbers to categorical features then get the dummy variables
    df["age_map"] = df["age_map"].replace(0, "working")
    df["age_map"] = df["age_map"].replace(1, "senior")

    df = pd.concat([df, pd.get_dummies(df.age_map,prefix='is')], axis=1)

    df.drop(["NumberOfOpenCreditLinesAndLoans",\
                     "NumberOfTimes90DaysLate","NumberRealEstateLoansOrLines","NumberOfTime60-89DaysPastDueNotWorse",\
                     "WithDependents","age_map","Unnamed: 0","is_senior","is_working", "MonthlyDebtPayments","SeriousDlqin2yrs"], 
            axis=1, inplace=True)
    predicted_data= hybrid_model.predict_proba(df)[:,1] * 100 
    predicted_data = np.round(predicted_data,2)

    return pd.concat([df,pd.DataFrame(predicted_data, columns = ["ChanceOfRisk in %"])], axis = 1)


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = pybase64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    return f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'

def ShowVisuals(df):
    fig, axs = plt.subplots(nrows=2, ncols=2,figsize=(14,11))

    df["Age_Group"] = df["age"]
    df.loc[(df["age"] >= 18) & (df["age"] < 60), "Age_Group"] = 0
    df.loc[(df["age"] >= 60), "Age_Group"] = 1

    #replacing those numbers to categorical features then get the dummy variables
    df["Age_Group"] = df["Age_Group"].replace(0, "working")
    df["Age_Group"] = df["Age_Group"].replace(1, "senior")


    sns.barplot(df["Age_Group"], df["SeriousDlqin2yrs"], ax= axs[0][0])
    sns.barplot(df["SeriousDlqin2yrs"], df["DebtRatio"], ax= axs[0][1])
    sns.barplot(df["SeriousDlqin2yrs"], df["RevolvingUtilizationOfUnsecuredLines"], ax= axs[1][0])
    sns.barplot(df["SeriousDlqin2yrs"], df["MonthlyIncome"], ax= axs[1][1])
    plt.show()
    
    
def main(): 

    st.image("nttlogo-black.png", width = 220)
    html_temp = """
    <div style="background-color:#ADD8E6;padding:5px">
    <h2 style="color:white;font-size:xx-large;text-align:center;text-shadow:2px 2px #808080;">Finance Distress Predictor ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    activities = ["Know The Risk","Know The Customers"]	
    choice = st.sidebar.selectbox("Select Activities",activities)
    
    if choice == 'Know The Risk':
    
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
                st.warning('{}% Chance of risk'.format(((p)*[100])))
            else:
                st.success('{}% Chance of risk'.format((p)*[100]))
                
                
    if choice == 'Know The Customers':
        #st.subheader("Exploratory Data Analysis")
        st.set_option('deprecation.showfileUploaderEncoding', False)
        data = st.file_uploader("Upload a Dataset", type=["csv", "txt"],showfileUploaderEncoding=False)
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")
            
            if st.button('Predict'):
                
                data = Preprocessing(df)
                st.dataframe(data.head())
                st.success("Sucessfully Predicted")
                
                st.markdown(get_table_download_link(data), unsafe_allow_html=True)
                
                
            if st.checkbox('Show Visuals'):
                st.write(ShowVisuals(df))
                st.pyplot()
                
         
    
if __name__=='__main__':
    main()
