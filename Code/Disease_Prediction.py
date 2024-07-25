from streamlit_option_menu import option_menu
import streamlit as st
from PIL import Image
import numpy as np
import pickle
import plotly.express as px
import pandas as pd
import seaborn as sns
import warnings
import tensorflow as tf
from PIL import Image
warnings.filterwarnings('ignore')
from apps.DiseaseModels import DiseaseModel
from apps.helper import prepare_symptoms_array
import joblib

st.set_page_config(page_title="NaVa HealthCare", page_icon="🧬", layout="wide")
st.title('🧬 NaVa HealthCare System : On Your Hand')
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)


st.markdown(
        """
        <div style='text-align: center; font-family: "Courier New", Courier, monospace;'>
            Welcome to your one-stop solution for medical predictions and analysis.
        </div>
        """,unsafe_allow_html=True
    )

st.markdown("---")
st.markdown("")
st.markdown("")
   
    
st.subheader('What can you do?')
st.markdown('''
           This web application is designed to analyse & predict disease. The system processes the symptoms provided by the user as input and gives the output as the probability of the disease.
        '''
    )
st.markdown("")
st.markdown("")
   
st.caption(
"<span style='font-size: small;'>Note: This app is not a substitute for professional medical advice.</span>\n\n"
"<span style='font-size: small;'>IF You feel You Can Always consult with a  professional for accurate diagnosis and treatment.</span>",
unsafe_allow_html=True
)



#Sidebar for navigation
st.sidebar.image(Image.open("D:/final_project/Code/image/Nava.png"))
st.sidebar.header("Choose according to your choices: ")



#Loading the saved models


diabetes_model = pickle.load(open('D:/final_project/models/diabetes_model.sav','rb'))
heart_disease_model = pickle.load(open('D:/final_project/Heart Disease Prediction/heart_disease_model.sav','rb'))
parkinsons_model = pickle.load(open('D:/final_project/Parkinsons Disease Prediction/parkinsons_model.sav','rb'))
Breast_Cancer_Prediction_model = pickle.load(open('D:/final_project/Breast_Cancer_Prediction/class.pkl','rb'))
Liver_Disease_models = pickle.load(open('D:/final_project/Liver Diesease Prediction/Liver_Disease_models.sav','rb'))
lung_cancer_model = joblib.load("D:/final_project/models/lung_cancer_model.sav")
hepatitis_model = joblib.load("D:/final_project/models/hepititisc_model.sav")

# load model
svc = pickle.load(open('D:/final_project/Personalized_Medicine_Recommending/model/svc.pkl','rb'))


# load databasedataset
sym_des = pd.read_csv("D:/final_project/Personalized_Medicine_Recommending/dataset/Symptom-severity.csv")
precautions = pd.read_csv("D:/final_project/Personalized_Medicine_Recommending/dataset/precautions_df.csv")
workout = pd.read_csv("D:/final_project/Personalized_Medicine_Recommending/dataset/workout_df.csv")
description = pd.read_csv("D:/final_project/Personalized_Medicine_Recommending/dataset/description.csv")
medications = pd.read_csv('D:/final_project/Personalized_Medicine_Recommending/dataset/medications.csv')
diets = pd.read_csv("D:/final_project/Personalized_Medicine_Recommending/dataset/diets.csv")




with st.sidebar:
    
    selected_page =option_menu('NaVa HealthCare',
                           ['Disease Prediction',
                            'Medicine Recommends',
                            ],
                           icons = ['activity','',],
                           default_index = 0)
    
    selected = option_menu('NaVa Prediction System',
                           ['Choose any---',
                            'Heart Disease',
                            'Liver Disease',
                            'Diabetes Disease',
                            'Parkinsons disease',
                            'Breast Disease',
                            'Skin Detection',
                            'Lung Detection',
                            ],
                           icons = ['','heart','person','activity','person','person','person','lungs'],
                           default_index = 0)



   

if selected_page == 'Medicine Recommends': 
    st.title('Personalized Medicine Recommends')
    def main(dis):
     # giving a title
    # st.title('Personalized Medicine Recommending')  
    
        desc = description[description['Disease'] == dis]['Description']
        desc = " ".join([w for w in desc])

        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
        pre = [col for col in pre.values]

        med = medications[medications['Disease'] == dis]['Medication']
        med = [med for med in med.values]

        die = diets[diets['Disease'] == dis]['Diet']
        die = [die for die in die.values]

        wrkout = workout[workout['disease'] == dis] ['workout']


        return desc,pre,med,die,wrkout    
    symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
    diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}  

# Model Prediction function
    def get_predicted_value(patient_symptoms):
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            input_vector[symptoms_dict[item]] = 1
        return diseases_list[svc.predict([input_vector])[0]]


    symptoms = st.text_input('Enter Your Symptoms Here (separated by this _)')
    st.text('Example likes chills, itching, high_fever, skin_rash')
    user_symptoms = [s.strip() for s in symptoms.split(',')]

    if st.button('Predict Disease'):
        if not all(symptom in symptoms_dict for symptom in user_symptoms):
            st.write("Invalid entry. Please check your symptoms and try again.")
        else:
            predicted_disease = get_predicted_value(user_symptoms)
            desc, pre, die, med, work = main(predicted_disease)

        # Displaying the prediction
            st.subheader('Predict Disease')
            st.write("Machine Prediction is:",predicted_disease,)
            st.subheader("Description")
            st.write(desc)
            st.subheader('Predicted Precaution')
            for i, j in enumerate(pre[0], 1):
                st.write(f"{i}: {j}")
            st.subheader('Predicted Diets')
            for i in die:
                st.write(i)
            st.subheader('Predicted Medication')
            for i, j in enumerate(med, 1):
                st.write(f"{i}: {j}")
            st.subheader('Predicted workout')
            for i, j in enumerate(work, 1):
                st.write(f"{i}: {j}")
    else:
        st.write('Click the button to predict disease.')
    




# multiple disease prediction
if selected_page == 'Disease Prediction': 
    # Create disease class and load ML model
    disease_model = DiseaseModel()
    disease_model.load_xgboost("D:/final_project/models/xgboost_model.json")

    # Title
    st.write('# Disease Prediction using Machine Learning')

    symptoms = st.multiselect('What are your symptoms?', options=disease_model.all_symptoms)

    X = prepare_symptoms_array(symptoms)

    # Trigger XGBoost model
    if st.button('Predict'): 
        
        # Run the model with the python script
        
        prediction, prob = disease_model.predict(X)
        st.write(f'## Disease: {prediction} with {prob*100:.2f}% probability')


        tab1, tab2= st.tabs(["Description", "Precautions"])

        with tab1:
            st.write(disease_model.describe_predicted_disease())

        with tab2:
            precautions = disease_model.predicted_disease_precautions()
            for i in range(4):
                st.write(f'{i+1}. {precautions[i]}')
    else:
        st.write('Click the button to predict disease.')





   
#Breast Disease Prediction Page

if(selected == 'Breast Disease'):
    
    #Page title
    st.title('Breast Disease Prediction')
    image = Image.open("D:/final_project/Code/image/bddss.png")
    st.image(image, caption='Breast Disease')
    name = st.text_input("Name:")
    
    
    
    diagnosis = st.selectbox('The diagnosis of breast tissues (M = Malignant: 0, B = Benign: 1)',[0,1])
    radius_mean = st.number_input("Enter your Radius Mean")
    texture_mean = st.number_input("Enter your Texture Mean")
    perimeter_mean = st.number_input("Enter your Perimeter Mean")
    area_mean = st.number_input("Enter your Area Mean")
    smoothness_mean = st.number_input("Enter your Smoothness Mean")
    compactness_mean = st.number_input("Enter your Compactness Mean")
    concavity_mean = st.number_input("Enter your Concavity Mean")
    concave_points_mean = st.number_input("Enter your Concave Points Mean")
    symmetry_mean = st.number_input("Enter your Symmetry Mean")
    fractal_dimension_mean = st.number_input("Enter your Fractal Dimension Mean")
    radius_se = st.number_input("Enter your Radius SE")
    texture_se = st.number_input("Enter your Texture SE")
    perimeter_se = st.number_input("Enter your Perimeter SE")
    area_se = st.number_input("Enter your Area SE")
    smoothness_se = st.number_input("Enter your Smoothness SE")
    compactness_se = st.number_input("Enter your Compactness SE")
    concavity_se = st.number_input("Enter your Concavity SE")
    concave_points_se = st.number_input("Enter your Concave Points SE")
    symmetry_se = st.number_input("Enter your Symmetry SE")
    fractal_dimension_se = st.number_input("Enter your Fractal Dimension SE")
    radius_worst = st.number_input("Enter your Radius Worst")
    texture_worst = st.number_input("Enter your Texture Worst")
    perimeter_worst = st.number_input("Enter your Perimeter Worst")
    area_worst = st.number_input("Enter your Area Worst")
    smoothness_worst = st.number_input("Enter your Smoothness Worst")
    compactness_worst = st.number_input("Enter your Compactness Worst")
    concavity_worst = st.number_input("Enter your Concavity Worst")
    concave_points_worst = st.number_input("Enter your Concave Points Worst")
    symmetry_worst = st.number_input("Enter your Symmetry Worst")
    fractal_dimension_worst = st.number_input("Enter your Fractal Dimension Worst")

    
    
    
    #Code for prediction
    Breast_Cancer_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Breast Disease Test Result'):
        Breast_Cancer_prediction = Breast_Cancer_Prediction_model.predict([[radius_mean,texture_mean,perimeter_mean,area_mean,smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean,fractal_dimension_mean,radius_se,texture_se,perimeter_se,area_se,smoothness_se,compactness_se,concavity_se,concave_points_se,symmetry_se,fractal_dimension_se,radius_worst,texture_worst,perimeter_worst,area_worst,smoothness_worst,compactness_worst,concavity_worst,concave_points_worst,symmetry_worst,fractal_dimension_worst]])
        
        if (Breast_Cancer_prediction[0]==1):
            Breast_Cancer_diagnosis = "The model predicts that you have Breast Cancer, Consult With Specialist."
            image = Image.open('D:/final_project/Code/image/positive.jpg')
            st.image(image, caption='')
            
        else:
            Breast_Cancer_diagnosis = "The model predicts that you don't have Breast Cancer."
            image = Image.open('D:/final_project/Code/image/negative.jpg')
            st.image(image, caption='')
            
            
    st.success(name+' , ' + Breast_Cancer_diagnosis)




  
#Diabetes Prediction Page

if(selected == 'Diabetes Disease'):
    
    #Page title
    st.title('Diabetes Disease Prediction')
    image = Image.open('D:/final_project/Code/image/d3.jpg')
    st.image(image, caption='diabetes Disease')
    name = st.text_input("Name:")
    
    
    Glucose = st.number_input('Glucose Level')
    BloodPressure = st.number_input('Blood Pressure value')
    SkinThickness = st.number_input('Skin Thickness value')
    Insulin = st.number_input('Insulin Level')
    BMI = st.number_input('BMI value')
    DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value')
    Age = st.number_input('Age of the Person')
    Pregnancies = st.slider('Number of Pregnancies):', min_value=0, max_value=4)
    
    #Code for prediction
    diabetes_dig = ''
    
    #Creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(
            [[Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age,Pregnancies]])
        
        if (diab_prediction[0]==1):
            diabetes_dig = "we are really sorry to say but it seems like you are Diabetic, Consult With Specialist."
            image = Image.open('D:/final_project/Code/image/positive.jpg')
            st.image(image, caption='')
            
        else:
            diabetes_dig = 'Congratulation,You are not diabetic'
            image = Image.open('D:/final_project/Code/image/negative.jpg')
            st.image(image, caption='')
            
            
    st.success(name+' , ' + diabetes_dig)
    
    
    
            
#Heart Disease Prediction Page
if(selected == 'Heart Disease'):
    
    #Page title
    st.title('Heart Disease Prediction')
    image = Image.open('D:/final_project/Code/image/heart2.jpg')
    st.image(image, caption='heart Disease')
    name = st.text_input("Name:")
    
    
    age = st.number_input('Age of the Person')
    sex = st.selectbox("Select Gender ( female:0, male:1 )",[0,1])
    cp = st.selectbox('Chest pain types ( normal:0, mid:1, high:2 )',[0,1,2,])
    trestbps = st.number_input('Resting Blood Pressure')
    chol = st.number_input('Serum Cholestoral in mg/dl')
    fbs=st.selectbox("Fasting Blood Sugar higher than 120 mg/dl(Yes:0,'No:1)",[0,1])
    restecg=st.selectbox("Resting Electrocardiographic Results (Nothing to note:0, ST-T Wave abnormality:1, Possible or definite left ventricular hypertrophy:2 )",[0,1,2,])
    thalach = st.number_input('Maximum Heart Rate achieved')
    exang=st.selectbox('Exercise Induced Angina (Yes:0,No:1)',[0,1])
    oldpeak = st.number_input('ST depression induced by exercise relative to rest')
    slope = st.selectbox('Heart Rate Slope (Upsloping: better heart rate with excercise(uncommon):0 ,Flatsloping: minimal change(typical healthy heart):1, Downsloping: signs of unhealthy heart:2)',[0,1,2])
    ca=st.selectbox('Number of Major Vessels Colored by Flourosopy',range(0,5,1))
    thal = st.selectbox('Thalium Stress Result (normal:0, fixed defect:1, reversable defect:2)',[0,1,2])

    
    #Code for prediction
    heart_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Heart Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if (heart_prediction[0]==1):
            heart_diagnosis = 'we are really sorry to say but it seems like you have Heart Disease, Consult With Specialist.'
            image = Image.open('D:/final_project/Code/image/positive.jpg')
            st.image(image, caption='')
            
        else:
            heart_diagnosis = "Congratulation , You don't have Heart Disease."
            image = Image.open('D:/final_project/Code/image/negative.jpg')
            st.image(image, caption='')
            
            
    st.success(name+' , ' + heart_diagnosis)
    
 
 
    
#Liver Disease Prediction Page
if(selected == 'Liver Disease'):
    
    #Page title
    st.title('Liver Disease Prediction')
    image = Image.open("D:/final_project/Code/image/liver.jpg")
    st.image(image, caption='Liver Disease.')
    name = st.text_input("Name:")
    
    
    Age = st.number_input('Entre your Age of the Person')
    Gender = st.selectbox('Entre your Gender of the Person( Female:0, male:1 )',[0,1,])
    Total_Bilirubin = st.number_input('Entre your Total Billirubin in mg/dL')
    Direct_Bilirubin = st.number_input('Entre your Conjugated Billirubin in mg/dL')
    Alkaline_Phosphotase = st.number_input('Entre your Alkaline Phosphotase in IU/L')
    Alamine_Aminotransferase = st.number_input('Entre your Alamine Aminotransferase in IU/L')
    Aspartate_Aminotransferase = st.number_input('Entre your Aspartate Aminotransferase in IU/L')
    Total_Protiens = st.number_input('Entre your Total_Protiens in g/dL ')
    Albumin = st.number_input('Entre your Albumin in g/dL')
    Albumin_and_Globulin_Ratio = st.number_input('Albumin and Globulin Ratio')
    Dataset = st.selectbox('Pain (Normal:0, Middle:1, High:2)',[0,1,2])

    #Code for prediction
    Liver_Disease_diagnosis = ''
    
    #Creating a button for prediction
    
    if st.button('Liver Disease Test Result'):
        Liver_Disease_prediction = Liver_Disease_models.predict([[Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Dataset]])
        
        if Liver_Disease_prediction[0] == 1:
            image = Image.open('D:/final_project/Code/image/positive.jpg')
            st.image(image, caption='')
            Liver_Disease_diagnosis = "we are really sorry to say but it seems like you have liver disease, Consult With Specialist."
        else:
            image = Image.open('D:/final_project/Code/image/negative.jpg')
            st.image(image, caption='')
            Liver_Disease_diagnosis = "Congratulation , You don't have liver disease."
    st.success(name+' , ' + Liver_Disease_diagnosis)
  

    
#Parkinsons Prediction Page
if(selected == 'Parkinsons disease'):
    
    #Page title
    st.title('Parkinsons Disease Prediction')
    image = Image.open("D:/final_project/Code/image/p1.jpg")
    st.image(image, caption='parkinsons disease')
    name = st.text_input("Name:")
    

    fo = st.number_input('MDVP Fo:(Hz)')
    fhi = st.number_input('MDVP Fhi:(Hz)')
    flo = st.number_input('MDVP Flo:(Hz)')
    Jitter_percent = st.number_input('MDVP:Jitter(%)')
    Jitter_Abs = st.number_input('MDVP:Jitter(Abs)')
    RAP = st.number_input('MDVP:RAP')
    PPQ = st.number_input('MDVP:PPQ')
    DDP = st.number_input('Jitter:DDP')
    Shimmer = st.number_input('MDVP:Shimmer')
    Shimmer_dB = st.number_input('MDVP:Shimmer(dB)')
    APQ3 = st.number_input('Shimmer:APQ3')
    APQ5 = st.number_input('Shimmer:APQ5')
    APQ = st.number_input('MDVP:APQ')
    DDA = st.number_input('Shimmer:DDA')
    NHR = st.number_input('NHR')
    HNR = st.number_input('HNR')
    RPDE = st.number_input('RPDE')
    DFA = st.number_input('DFA')
    spread1 = st.number_input('spread1')
    spread2 = st.number_input('spread2')
    D2 = st.number_input('D2')
    PPE = st.number_input('PPE')
        
        
    #Code for prediction
    parkinsons_diagnosis = ''
        
    #Creating a button for prediction
        
    if st.button('Parkinsons Test Result'):
            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])
            
            if (parkinsons_prediction[0]==1):
                parkinsons_diagnosis = 'we are really sorry to say but it seems like you have Parkinson disease, Consult With Specialist.'
                image = Image.open('D:/final_project/Code/image/positive.jpg')
                st.image(image, caption='')
                
            else:
                parkinsons_diagnosis = "Congratulation , You don't have Parkinson disease"
                image = Image.open('D:/final_project/Code/image/negative.jpg')
                st.image(image, caption='')
                
                
    st.success(name+' , ' + parkinsons_diagnosis)




# Lung Cancer prediction page
# Load the dataset
lung_cancer_data = pd.read_csv("D:/final_project/Datasets/survey lung cancer.csv")

# Convert 'M' to 0 and 'F' to 1 in the 'GENDER' column
lung_cancer_data['GENDER'] = lung_cancer_data['GENDER'].map({'M': 'Male', 'F': 'Female'})


if selected == 'Lung Detection':
    st.title("Lung Disease Prediction")
    image = Image.open("D:/final_project/Code/image/h.png")
    st.image(image, caption='Lung Disease')

    # Columns
    # No inputs from the user
    name = st.text_input("Name:")
    gender = st.selectbox("Gender:", lung_cancer_data['GENDER'].unique())
    age = st.number_input("Age")
    smoking = st.selectbox("Smoking:", ['NO', 'YES'])
    yellow_fingers = st.selectbox("Yellow Fingers:", ['NO', 'YES'])
    anxiety = st.selectbox("Anxiety:", ['NO', 'YES'])
    peer_pressure = st.selectbox("Peer Pressure:", ['NO', 'YES'])
    chronic_disease = st.selectbox("Chronic Disease:", ['NO', 'YES'])
    fatigue = st.selectbox("Fatigue:", ['NO', 'YES'])
    allergy = st.selectbox("Allergy:", ['NO', 'YES'])
    wheezing = st.selectbox("Wheezing:", ['NO', 'YES'])
    alcohol_consuming = st.selectbox("Alcohol Consuming:", ['NO', 'YES'])
    coughing = st.selectbox("Coughing:", ['NO', 'YES'])
    shortness_of_breath = st.selectbox("Shortness of Breath:", ['NO', 'YES'])
    swallowing_difficulty = st.selectbox("Swallowing Difficulty:", ['NO', 'YES'])
    chest_pain = st.selectbox("Chest Pain:", ['NO', 'YES'])

    # Code for prediction
    Lung_result = ''

    # Button
    if st.button("Lung Test Result"):
        # Create a DataFrame with user inputs
        user_data = pd.DataFrame({
            'GENDER': [gender],
            'AGE': [age],
            'SMOKING': [smoking],
            'YELLOW_FINGERS': [yellow_fingers],
            'ANXIETY': [anxiety],
            'PEER_PRESSURE': [peer_pressure],
            'CHRONICDISEASE': [chronic_disease],
            'FATIGUE': [fatigue],
            'ALLERGY': [allergy],
            'WHEEZING': [wheezing],
            'ALCOHOLCONSUMING': [alcohol_consuming],
            'COUGHING': [coughing],
            'SHORTNESSOFBREATH': [shortness_of_breath],
            'SWALLOWINGDIFFICULTY': [swallowing_difficulty],
            'CHESTPAIN': [chest_pain]
        })

        # Map string values to numeric
        user_data.replace({'NO': 1, 'YES': 2}, inplace=True)

        # Strip leading and trailing whitespaces from column names
        user_data.columns = user_data.columns.str.strip()

        # Convert columns to numeric where necessary
        numeric_columns = ['AGE', 'FATIGUE', 'ALLERGY', 'ALCOHOLCONSUMING', 'COUGHING', 'SHORTNESSOFBREATH']
        user_data[numeric_columns] = user_data[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Perform prediction
        cancer_prediction = lung_cancer_model.predict(user_data)

        # Display result
        if cancer_prediction[0] == 'YES':
            Lung_result  = "The model predicts that there is a risk of Lung Cancer, Consult With Specialist."
            image = Image.open('D:/final_project/Code/image/positive.jpg')
            st.image(image, caption='')
        else:
            Lung_result  = "The model predicts no significant risk of Lung Cancer."
            image = Image.open('D:/final_project/Code/image/negative.jpg')
            st.image(image, caption='')

        st.success(name + ', ' + Lung_result )





   
#Skin Detection Disease Prediction Page
if(selected == 'Skin Detection'):

    interpreter_skin = tf.lite.Interpreter(model_path='D:/final_project/models/skinmodel.tflite')
    interpreter_skin.allocate_tensors()
    input_details_skin = interpreter_skin.get_input_details()
    output_details_skin = interpreter_skin.get_output_details()

    # Map class indices to class names
    class_names = {
        0: 'Melanocytic nevi',
        1: 'Melanoma',
        2: 'Benign keratosis-like lesions',
        3: 'Basal cell carcinoma',
        4: 'Actinic keratoses',
        5: 'Vascular lesions',
        6: 'Dermatofibroma'
    }

    st.subheader("Skin Disease Classification", divider='grey')
    image = Image.open("D:/final_project/image/skin-disease-different-diseasessss.png")
    st.image(image, caption='Skin Disease')
    st.markdown("")
    st.caption("Upload a Dermatocopic Image of Skin Lesion")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=False)

        img_array = np.asarray(image.resize((100, 75)))
        img_array = img_array.astype(np.float32) / 255.0  
        img_array = np.expand_dims(img_array, axis=0)  

        interpreter_skin.set_tensor(input_details_skin[0]['index'], img_array)
        interpreter_skin.invoke()
        prediction = interpreter_skin.get_tensor(output_details_skin[0]['index'])
        predicted_class = np.argmax(prediction)

        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence: {prediction[0][predicted_class]:.4f}")


    
# Ending 
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<div style="font-size: 24px; color: #9370DB; text-align: center;">Hope to see you soon :)</div>', unsafe_allow_html=True)
st.image(Image.open("D:/final_project/Code/image/Nava.png"), caption="Your Health, Our Priority")
st.sidebar.markdown("---")
st.sidebar.markdown("made by -- [MD Nayab Ansari](https://github.com/nayuansari)")



    


