import pandas as pd 
import numpy as np                     # For mathematical calculations 
import seaborn as sns                  # For data visualization 
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk
import os
import joblib

@st.cache
def load_data(dataset):
	df = pd.read_csv(dataset)
	return df


def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


gender_label = {'Male': 1, 'Female': 0}
married_label = {'Yes': 0, 'No': 1}
education_label = {'Educated': 1, 'Not Educated': 0}
employment_label = {'Self Employment': 0, 'Not Self Employment': 1}
dependents_label = {'0': 0, '1': 1, '2': 2, '3+': 3}
history_label = {'Yes' : 1 , 'No': 0}

properties_label = {'Semi Urban': 1, 'Urban': 2, 'Rural': 3}
class_label = {'good': 0, 'acceptable': 1, 'very good': 2, 'unacceptable': 3}

# Get the Keys
def get_value(val,my_dict):
	for key ,value in my_dict.items():
		if val == key:
			return value

# Find the Key From Dictionary
def get_key(val,my_dict):
	for key ,value in my_dict.items():
		if val == value:
			return key



# """Loan approval with ML Streamlit App"""

st.title("Loan approval")
st.subheader("Check out whether loan can be given or not")
st.subheader("Streamlit ML App")
# st.image(load_image("cars_images/car1.jpg"),width=300, caption='Images')

activities = ['EDA','Prediction','Gallery','About']
choices = st.sidebar.selectbox("Select Activity",activities)

if choices == 'EDA':
	st.subheader("EDA")
	if st.checkbox('Show Data'):
		data = load_data('train_ctrUa4K.csv')
		st.dataframe(data.head(5))

	if st.checkbox("Show Summary of Dataset"):
		st.write(data.describe())

		# Show Plots
	if st.checkbox("Simple Value Plots "):
		st.write(sns.countplot(data['Loan_Status']))
			# Use Matplotlib to render seaborn
		st.pyplot()

		# Show Columns By Selection
	if st.checkbox("Select Columns To Show"):
		all_columns = data.columns.tolist()
		selected_columns = st.multiselect('Select',all_columns)
		new_df = data[selected_columns]
		st.dataframe(new_df)

	if st.checkbox("Pie Plot"):
		all_columns_names = data.columns.tolist()
		if st.button("Generate Pie Plot"):
			st.write(data.iloc[:,-1].value_counts().plot.pie(autopct="%1.1f%%"))
			st.pyplot()

if choices == 'Prediction':
		st.subheader("Prediction")
		
		married = st.selectbox('Marital Status',tuple(married_label.keys()))
		employment = st.selectbox("Employment",tuple(employment_label.keys()))
		education = st.selectbox('Education',tuple(education_label.keys()))
		gender = st.selectbox('Select your Gender',tuple(gender_label.keys()))
		dependents = st.selectbox('Number of dependents', tuple(dependents_label.keys()))
		Income = st.slider('Your income', 0 ,10000)
		Coincome = st.slider('Your nominee income', 0 ,10000)
		Amount=st.slider('Loan amount',0,10000)
		term=st.selectbox('Term of loan in days',[30,90,180,360])
		history = st.selectbox('Credit History',tuple(history_label.keys()))
		properties = st.selectbox('Area of Transaction',tuple(properties_label.keys()))

		k_gender = get_value(gender,gender_label)
		k_married = get_value(married,married_label)
		k_education = get_value(education,education_label)
		k_employment = get_value(employment,employment_label)
		k_dependents = get_value(dependents,dependents_label)
		k_history = get_value(history,history_label)
		k_properties = get_value(properties,properties_label)

		

		
		# pretty_data = {
		# "buying":buying,
		# "maint":maint,
		# "doors":doors,
		# "persons":persons,
		# "lug_boot":lug_boot,
		# "safety":safety,
		# }
		# st.subheader("Options Selected")
		# st.json(pretty_data)

		st.subheader("Data Encoded As")
		# Data To Be Used
		sample_data = [k_married,k_employment,k_education,k_gender,k_dependents,Income,Coincome,Amount,term,k_history,k_properties]
		st.write(sample_data)

		prep_data = np.array(sample_data).reshape(1, -1)

		# model_choice = st.selectbox("Model Type",['logit','naive bayes','MLP classifier'])
		if st.button('Evaluate'):
			predictor = load_prediction_models("models/logit_model.pkl")
			prediction = predictor.predict(prep_data)
			st.write(prediction)

			
			final_result = get_key(prediction,class_label)
			st.success(final_result)



 









