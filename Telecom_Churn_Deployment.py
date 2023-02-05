import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import streamlit as st

st.title('Telecom Churn Prediction')
st.sidebar.header("'1' means YES and '0' means NO ")
st.sidebar.header('User Input Parameters')

df=pd.read_csv("Churn.csv")
df=df.drop(columns={'Unnamed: 0','state','area.code'},axis=1)
df.rename(columns={'account.length': 'Actv_Days', 'voice.plan': 'voice_plan',
                       'voice.messages':'voice_msgs', 'intl.plan':'intl_plan',
                       'intl.mins':'intl_mins','intl.calls':'intl_calls',
                       'intl.charge':'intl_chrg','day.mins':'day_mins',
                       'day.calls':'day_calls','day.charge':'day_chrg','eve.mins':'eve_mins',
                       'eve.calls':'eve_calls','eve.charge':'eve_chrg','night.mins':'night_mins','night.calls':'night_calls',
                       'night.charge':'night_chrg','customer.calls':'cust_calls'},inplace=True)


# One-hot encoding

df['voice_plan'].replace(('yes', 'no'), (1, 0), inplace=True)
df['intl_plan'].replace(('yes', 'no'), (1, 0), inplace=True)
df['churn'].replace(('yes', 'no'), (1, 0), inplace=True)
for col in ['day_chrg', 'eve_mins']:
    df[col] = df[col].astype('float64')


#Data Imputation
df['day_chrg'] = df['day_chrg'].fillna(df['day_chrg'].mean())
df['eve_mins'] = df['eve_mins'].fillna(df['eve_mins'].mean())

def user_input_features():
    Actv_Days = st.sidebar.number_input("Number of days the customer is with company")
    V_plan = st.sidebar.selectbox(" Voice Plan " ,('1','0'))
    V_msgs = st.sidebar.number_input("Number of vociemail messages")
    intl_plan = st.sidebar.selectbox(" International Plan",('1','0'))
    intl_calls = st.sidebar.number_input("Total number of international calls")
    intl_chrg = st.sidebar.number_input("Total charge for international calls")
    day_calls = st.sidebar.number_input("Total number of calls in day")
    day_chrg = st.sidebar.number_input("Total charge in the day")
    eve_calls = st.sidebar.number_input("Total number of calls in evening")
    eve_chrg = st.sidebar.number_input("Total charge in the evening")
    n_calls = st.sidebar.number_input(" Total number of calls during the night")
    n_chrg = st.sidebar.number_input("Total charge during the night")
    cust_calls = st.sidebar.number_input("Number of calls to customer service")
    data = {'Actv_Days':Actv_Days,
            'voice_plan': V_plan,
            'voice_msgs': V_msgs,
            'intl_plan': intl_plan,
            'intl_calls':intl_calls,
            'intl_chrg':intl_chrg,
            'day_calls':day_calls,
            'day_chrg':day_chrg,
            'eve_calls':eve_calls,
            'eve_chrg':eve_chrg,
            'night_calls':n_calls,
            'night_chrg':n_chrg,
            'cust_calls':cust_calls}
    features = pd.DataFrame(data, index=[0])
    return features

df_ip = user_input_features()
st.subheader("User input parameters")
st.write(df_ip)

#droping the columns which have high correlated features
df=df.drop(["intl_mins","day_mins","eve_mins","night_mins"],axis=1)

# Removing the Outliers from the dataframe which have 'NO' Churn Value
l1 = ['Actv_Days', 'voice_msgs','intl_calls','intl_chrg','day_calls','day_chrg','eve_calls','eve_chrg','night_calls','night_chrg','cust_calls']
for i in l1:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1
    for j in range(len(df)):
        if j not in df.index:
            continue
        if df['churn'][j] == 0:
            if (df[i][j] < (Q1 - (1.5*IQR))) or (df[i][j] > (Q3 + (1.5*IQR))):
                df=df.drop(j)

#Scaling Data

mms = MinMaxScaler()
df[['Actv_Days','voice_msgs','intl_calls', 'intl_chrg', 'day_calls', 
    'day_chrg', 'eve_calls', 'eve_chrg',  'night_calls',
    'night_chrg', 'cust_calls']] = mms.fit_transform(df[['Actv_Days','voice_msgs','intl_calls', 'intl_chrg',  'day_calls',
                                                         'day_chrg', 'eve_calls', 'eve_chrg',  'night_calls',
                                                         'night_chrg', 'cust_calls']])
#Spliting Target and Independent Variable
X = df.drop(columns=['churn'])
Y = df['churn']
(X_train,X_test,y_train,y_test)=train_test_split(X,Y, test_size=0.2,stratify=Y, random_state=21)
df_test=pd.concat([X_test,y_test],axis=1)
#df_test.to_csv("Test_Data.csv")

#Final Model
GBR=GradientBoostingClassifier(learning_rate=0.1,n_estimators=100)

GBR.fit(X_train,y_train)

#Evaluation Metrics
y_pred = GBR.predict(X_test)
acc= accuracy_score(y_test,y_pred)*100

#st.write("Accuracy of the model :",acc)

#Model Prediction
prediction = GBR.predict(df_ip)
prediction_proba = GBR.predict_proba(df_ip)

st.write("The Prediction value is ",prediction[0])
st.subheader("Predicted Result")
st.write(" 'Yes' the customer will churn " if prediction==1 else " The customer will NOT churn ")



