import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
df = pd.read_csv("dataset.csv")

# Headings
st.title('Diabetes Checkup')
st.sidebar.header('Patient Data')
st.subheader('Training Data Stats')
st.write(df.describe())

# X and Y Data
x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Function to get user input
def user_report():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    bp = st.sidebar.slider('BloodPressure', 0, 122, 70)
    skinthickness = st.sidebar.slider('SkinThickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0, 67, 20)
    dpf = st.sidebar.slider('DiabetesPedigreeFunction', 0.0, 2.4, 0.47)
    age = st.sidebar.slider('Age', 21, 88, 33)

    user_report_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

# Patient Data
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

# Ensure the features match
user_data = user_data[x.columns]

# Model
rf = RandomForestClassifier()
rf.fit(x_train, y_train)
user_result = rf.predict(user_data)

# Prediction Output
st.subheader('Your Report: ')
output = 'You are not Diabetic' if user_result[0] == 0 else 'You are Diabetic'
st.title(output)
st.subheader('Accuracy: ')
st.write(f'{accuracy_score(y_test, rf.predict(x_test)) * 100:.2f}%')

# Visualization in Pie Chart Form
st.subheader('Visualisation')

# Plotting the pie chart for user's data
labels = user_data.columns.tolist()
sizes = user_data.iloc[0].tolist()

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)


