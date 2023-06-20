import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv(r"C:\REAL TIME PROJECTS\DIABETES PRIDECTION\diabetes_binary_health_indicators_BRFSS2015.csv")

data = data.drop(["Education"], axis = 1)
data = data.drop(["Income"], axis = 1)

st.title("DIABETES CHECKUP")

x = data.drop(["Diabetes_binary"], axis = 1)
y = data.iloc[:, 1]

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

def user_report():
  HighBP = st.sidebar.slider('HighBP',0,1)
  HighChol = st.sidebar.slider('HighChol',0,1)
  CholCheck = st.sidebar.slider('CholCheck',0,1)
  BMI = st.sidebar.slider('BMI',10,70)
  Smoker = st.sidebar.slider('Smoker',0,1)
  Stroke = st.sidebar.slider('Stroke',0,1)
  HeartDiseaseorAttack = st.sidebar.slider('HeartDiseaseorAttack',0,1)
  PhysActivity = st.sidebar.slider('PhysActivity',0,1)
  Fruits = st.sidebar.slider('Fruits',0,1)
  Veggies = st.sidebar.slider('Veggies',0,1)
  HvyAlcoholConsump = st.sidebar.slider('HvyAlcoholConsump',0,1)
  AnyHealthcare = st.sidebar.slider('AnyHealthcare',0,1)
  NoDocbcCost = st.sidebar.slider('NoDocbcCost',0,1)
  GenHlth = st.sidebar.slider('GenHlth',1,5)
  MentHlth = st.sidebar.slider('MentHlth',1,30)
  PhysHlth = st.sidebar.slider('PhysHlth',1,30)
  DiffWalk = st.sidebar.slider('DiffWalk',0,1)
  Sex = st.sidebar.slider('Sex',0,1)
  Age = st.sidebar.slider('Age',2,100)

  user_report = {

      'HighBP':HighBP,
      'HighChol':HighChol,
      'CholCheck':CholCheck,
      'BMI':BMI,
      'Smoker':Smoker,
      'Stroke':Stroke,
      'HeartDiseaseorAttack':HeartDiseaseorAttack,
      'PhysActivity':PhysActivity,
      'Fruits':Fruits,
      'Veggies':Veggies,
      'HvyAlcoholConsump':HvyAlcoholConsump,
      'AnyHealthcare':AnyHealthcare,
      'NoDocbcCost':NoDocbcCost,
      'GenHlth':GenHlth,
      'MentHlth':MentHlth,
      'PhysHlth':PhysHlth,
      'DiffWalk':DiffWalk,
      'Sex':Sex,
      'Age':Age  
  }
  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

user_data = user_report()

rf = RandomForestClassifier()
rf.fit(x_train,y_train)

user_result = rf.predict(user_data)

st.subheader("YOUR REPORT")

output = ''

if user_result[0] == 0:
  output = 'NOT DIABETIC'
else:
  output = 'MAY BE DIABETIC'

st.write(output)
