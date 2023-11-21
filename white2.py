# Open Sublime text editor, create a new Python file, copy the following code in it and save it as 'penguin_app.py'.

# Importing the necessary libraries.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression  
from sklearn.ensemble import RandomForestClassifier

# Load the DataFrame
csv_file = 'https://s3-student-datasets-bucket.whjr.online/whitehat-ds-datasets/penguin.csv'
df = pd.read_csv(csv_file)

# Display the first five rows of the DataFrame
df.head()

# Drop the NAN values
df = df.dropna()

# Add numeric column 'label' to resemble non numeric column 'species'
df['label'] = df['species'].map({'Adelie': 0, 'Chinstrap': 1, 'Gentoo':2})


# Convert the non-numeric column 'sex' to numeric in the DataFrame
df['sex'] = df['sex'].map({'Male':0,'Female':1})

# Convert the non-numeric column 'island' to numeric in the DataFrame
df['island'] = df['island'].map({'Biscoe': 0, 'Dream': 1, 'Torgersen':2})
island_dict = {'Biscoe': 0, 'Dream': 1, 'Torgersen':2}
sex_dict = {'Male':0,'Female':1}
# Create X and y variables
X = df[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex']]
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 42)


# Build a SVC model using the 'sklearn' module.
svc_model = SVC(kernel = 'linear')
svc_model.fit(X_train, y_train)
svc_score = svc_model.score(X_train, y_train)

# Build a LogisticRegression model using the 'sklearn' module.
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_score = log_reg.score(X_train, y_train)

# Build a RandomForestClassifier model using the 'sklearn' module.
rf_clf = RandomForestClassifier(n_jobs = -1)
rf_clf.fit(X_train, y_train)
rf_clf_score = rf_clf.score(X_train, y_train)
@st.cache
def prediction(model, island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex):
    result = model.predict([[island, bill_length_mm, bill_depth_mm, flipper_length_mm, body_mass_g, sex]])
    result = result[0]
    if result == 0:
        return 'Adelie'
    elif result == 1:
        return 'Chinstrap'
    else:
        return 'Gentoo'

st.sidebar.title('penguing  Species Prediction App')

island = st.sidebar.text_input('island')
bill_length_mm = st.sidebar.number_input('bill_length_mm')
bill_depth_mm = st.sidebar.number_input('bill_depth_mm')
flipper_length_mm = st.sidebar.number_input('flipper_length_mm')
body_mass_g = st.sidebar.number_input('body_mass_g')
sex = st.sidebar.text_input('Sex')

dropDownPicker = st.sidebar.selectbox('classifier',options=['SVC','RandomForestClassifier','LogisticRegression'])


if st.sidebar.button("Button Text"):
    if dropDownPicker == 'SVC':
        result = prediction(svc_model, island_dict[island], bill_length_mm, bill_depth_mm, flipper_length_mm,
                            body_mass_g, sex_dict[sex])
        st.write(result)
        st.write(svc_model.score(X_train, y_train))
    elif dropDownPicker == 'RandomForestClassifier':
        result = prediction(rf_clf, island_dict[island], bill_length_mm, bill_depth_mm, flipper_length_mm,
                            body_mass_g, sex_dict[sex])
        st.write(result)
        st.write(rf_clf.score(X_train, y_train))
    else:
        result = prediction(log_reg, island_dict[island], bill_length_mm, bill_depth_mm, flipper_length_mm,
                            body_mass_g, sex_dict[sex])
        st.write(result)
        st.write(log_reg.score(X_train, y_train))