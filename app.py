import numpy as np
import pandas as pd
import pickle
import streamlit as st
import ast


sys_des=pd.read_csv("./Datasets/symtoms_df.csv")
precautions=pd.read_csv("./Datasets/precautions_df.csv")
workout=pd.read_csv("./Datasets/workout_df.csv")
description=pd.read_csv("./Datasets/description.csv")
medications=pd.read_csv("./Datasets/medications.csv")
diets=pd.read_csv("./Datasets/diets.csv")

model=pickle.load(open('model.pkl', 'rb'))

def helper(disease):
    # Get description
    desc = description[description['Disease'] == disease]['Description']
    if not desc.empty:
        desc = " ".join([w for w in desc])
    else:
        desc = "No description available for this disease."
    
    # Get precautions
    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    if not pre.empty:
        pre = [col for col in pre.values]
    else:
        pre = ["No precautions available for this disease."]
    
    # Get medications
    med = medications[medications['Disease'] == disease]['Medication']
    if not med.empty:
        med = ", ".join(med.astype(str).values)
        med = med[1:-1]
        med = med.replace("'", "")
    else:
        med = "No medications available for this disease."
    
    # Get diet plan
    die = diets[diets['Disease'] == disease]['Diet']
    if not die.empty:
        die = die.values[0]
        die = ast.literal_eval(die)
        die = ', '.join(die)
    else:
        die = "No specific diet plan available for this disease."
    
    # Get workout recommendations
    wrkout = workout[workout['disease'] == disease]['workout']
    if not wrkout.empty:
        wrkout = wrkout.tolist()
    else:
        wrkout = ["No workout recommendations available for this disease."]
    
    return desc, pre, med, die, wrkout

symptoms_dict = {
    'Itching': 0, 'Skin Rash': 1, 'Nodal Skin Eruptions': 2, 'Continuous Sneezing': 3,
    'Shivering': 4, 'Chills': 5, 'Joint Pain': 6, 'Stomach Pain': 7, 'Acidity': 8,
    'Ulcers On Tongue': 9, 'Muscle Wasting': 10, 'Vomiting': 11, 'Burning Micturition': 12,
    'Spotting  Urination': 13, 'Fatigue': 14, 'Weight Gain': 15, 'Anxiety': 16,
    'Cold Hands And Feets': 17, 'Mood Swings': 18, 'Weight Loss': 19, 'Restlessness': 20,
    'Lethargy': 21, 'Patches In Throat': 22, 'Irregular Sugar Level': 23, 'Cough': 24,
    'High Fever': 25, 'Sunken Eyes': 26, 'Breathlessness': 27, 'Sweating': 28,
    'Dehydration': 29, 'Indigestion': 30, 'Headache': 31, 'Yellowish Skin': 32,
    'Dark Urine': 33, 'Nausea': 34, 'Loss Of Appetite': 35, 'Pain Behind The Eyes': 36,
    'Back Pain': 37, 'Constipation': 38, 'Abdominal Pain': 39, 'Diarrhoea': 40,
    'Mild Fever': 41, 'Yellow Urine': 42, 'Yellowing Of Eyes': 43, 'Acute Liver Failure': 44,
    'Fluid Overload': 45, 'Swelling Of Stomach': 46, 'Swelled Lymph Nodes': 47,
    'Malaise': 48, 'Blurred And Distorted Vision': 49, 'Phlegm': 50,
    'Throat Irritation': 51, 'Redness Of Eyes': 52, 'Sinus Pressure': 53, 'Runny Nose': 54,
    'Congestion': 55, 'Chest Pain': 56, 'Weakness In Limbs': 57, 'Fast Heart Rate': 58,
    'Pain During Bowel Movements': 59, 'Pain In Anal Region': 60, 'Bloody Stool': 61,
    'Irritation In Anus': 62, 'Neck Pain': 63, 'Dizziness': 64, 'Cramps': 65,
    'Bruising': 66, 'Obesity': 67, 'Swollen Legs': 68, 'Swollen Blood Vessels': 69,
    'Puffy Face And Eyes': 70, 'Enlarged Thyroid': 71, 'Brittle Nails': 72,
    'Swollen Extremeties': 73, 'Excessive Hunger': 74, 'Extra Marital Contacts': 75,
    'Drying And Tingling Lips': 76, 'Slurred Speech': 77, 'Knee Pain': 78,
    'Hip Joint Pain': 79, 'Muscle Weakness': 80, 'Stiff Neck': 81, 'Swelling Joints': 82,
    'Movement Stiffness': 83, 'Spinning Movements': 84, 'Loss Of Balance': 85,
    'Unsteadiness': 86, 'Weakness Of One Body Side': 87, 'Loss Of Smell': 88,
    'Bladder Discomfort': 89, 'Foul Smell Of Urine': 90, 'Continuous Feel Of Urine': 91,
    'Passage Of Gases': 92, 'Internal Itching': 93, 'Toxic Look (Typhos)': 94,
    'Depression': 95, 'Irritability': 96, 'Muscle Pain': 97, 'Altered Sensorium': 98,
    'Red Spots Over Body': 99, 'Belly Pain': 100, 'Abnormal Menstruation': 101,
    'Dischromic  Patches': 102, 'Watering From Eyes': 103, 'Increased Appetite': 104,
    'Polyuria': 105, 'Family History': 106, 'Mucoid Sputum': 107, 'Rusty Sputum': 108,
    'Lack Of Concentration': 109, 'Visual Disturbances': 110,
    'Receiving Blood Transfusion': 111, 'Receiving Unsterile Injections': 112,
    'Coma': 113, 'Stomach Bleeding': 114, 'Distention Of Abdomen': 115,
    'History Of Alcohol Consumption': 116, 'Fluid Overload.1': 117,
    'Blood In Sputum': 118, 'Prominent Veins On Calf': 119, 'Palpitations': 120,
    'Painful Walking': 121, 'Pus Filled Pimples': 122, 'Blackheads': 123,
    'Scurring': 124, 'Skin Peeling': 125, 'Silver Like Dusting': 126,
    'Small Dents In Nails': 127, 'Inflammatory Nails': 128, 'Blister': 129,
    'Red Sore Around Nose': 130, 'Yellow Crust Ooze': 131
}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    # Predict the class label (just the disease)
    predicted_class = model.predict([input_vector])[0]

    # Get the disease name from the predicted class
    predicted_disease = diseases_list[predicted_class]

    return predicted_disease


st.set_page_config(page_title="MedAI", page_icon="🩺")

# Title and Subtitle
st.markdown("<h1 style='color:#4B8BBE;'>🩺 MedAI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='color:gray;'>Smarter Care, Anytime, Anywhere.</h4>", unsafe_allow_html=True)

# Input Section
st.markdown("---")
st.markdown("### 🔍 Select Symptoms")
symptoms = st.multiselect("Choose your symptoms", sorted(list(symptoms_dict.keys())))

# Button
diagnosis = st.button("🧬 Get Diagnosis")

# Output Section
if diagnosis:
    if len(symptoms) == 0:
        st.warning("⚠️ Please select at least one symptom.")
    else:
        disease = get_predicted_value(symptoms)
        desc, pre, med, die, wrkout = helper(disease)

        # Create a collapsible container for the report
        st.expander("📝 Diagnosis Report", expanded=True)
        st.success(f"**Predicted Disease:** {disease}")

        # Description Section
        st.expander("🧾 Description", expanded=False).markdown(desc)

        # Medication Section
        st.expander("💊 Medication", expanded=False).markdown(f"<ul>{''.join(f'<li>{m}</li>' for m in med.split(', '))}</ul>", unsafe_allow_html=True)

        # Diet Plan Section
        st.expander("🥗 Diet Plan", expanded=False).markdown(f"<ul>{''.join(f'<li>{d}</li>' for d in die.split(', '))}</ul>", unsafe_allow_html=True)

        # Workout Recommendations Section
        st.expander("🏃 Workout Recommendations", expanded=False).markdown('\n'.join([f"- {tip}" for tip in wrkout]))

        # Footer (optional)
        st.markdown("---")
        st.markdown("<p style='text-align:center; color:gray;'>Powered by MedAI © 2025</p>", unsafe_allow_html=True)