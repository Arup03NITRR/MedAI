from fastapi import FastAPI, Request, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import pickle
import ast
from fpdf import FPDF
from datetime import datetime
import pytz
import io
import uvicorn

# Load datasets and model
sys_des = pd.read_csv("./Datasets/symtoms_df.csv")
precautions = pd.read_csv("./Datasets/precautions_df.csv")
workout = pd.read_csv("./Datasets/workout_df.csv")
description = pd.read_csv("./Datasets/description.csv")
medications = pd.read_csv("./Datasets/medications.csv")
diets = pd.read_csv("./Datasets/diets.csv")
doctor = pd.read_csv("./Datasets/disease_doctor.csv")
model = pickle.load(open('model.pkl', 'rb'))

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

app = FastAPI(title="MedAI Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 👈 Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PatientInput(BaseModel):
    name: str
    age: str
    gender: str
    phone: str
    email: str
    symptoms: list[str]


def helper(disease):
    desc = description[description['Disease'] == disease]['Description']
    desc = " ".join(desc.values) if not desc.empty else "No description available for this disease."

    pre = precautions[precautions['Disease'] == disease][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.tolist()[0] if not pre.empty else ["No precautions available for this disease."]

    med = medications[medications['Disease'] == disease]['Medication']
    med = ", ".join(med.astype(str).values) if not med.empty else "No medications available for this disease."

    die = diets[diets['Disease'] == disease]['Diet']
    die = ', '.join(ast.literal_eval(die.values[0])) if not die.empty else "No specific diet plan available."

    wrkout = workout[workout['disease'] == disease]['workout']
    wrkout = wrkout.tolist() if not wrkout.empty else ["No workout recommendations available."]

    return desc, pre, med, die, wrkout


def get_predicted_value(symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1
    pred_class = model.predict([input_vector])[0]
    return diseases_list.get(pred_class, "Unknown Disease")


def generate_patient_pdf(data: PatientInput, disease, desc, med, die, wrkout):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 15, "MedAI - Medical Report", ln=True, align="C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, "Smarter Care, Anytime, Anywhere.", ln=True, align="C")

    # Timestamp
    pdf.set_font("Arial", "B", 8)
    india_time = datetime.now(pytz.timezone('Asia/Kolkata'))
    pdf.cell(0, 8, india_time.strftime('%Y-%m-%d %H:%M:%S'), ln=True, align="R")

    # Patient Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Patient Details:", ln=True)
    pdf.set_font("Arial", "", 10)
    for label, value in data.dict().items():
        if label != "symptoms":
            pdf.multi_cell(0, 5, f"{label.capitalize()}: {value}")
    pdf.multi_cell(0, 5, f"Symptoms: {', '.join(data.symptoms)}")

    # Disease Info
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Diagnosis:", ln=True)
    pdf.set_font("Arial", "", 10)
    pdf.multi_cell(0, 5, f"Predicted Disease: {disease}")
    pdf.multi_cell(0, 5, f"Description: {desc}")
    pdf.multi_cell(0, 5, f"Medications: {med}")
    pdf.multi_cell(0, 5, "Diet Plan:")
    for d in die.split(','):
        pdf.multi_cell(0, 5, f"  - {d.strip()}")

    pdf.multi_cell(0, 5, "Workout Recommendations:")
    for w in wrkout:
        pdf.multi_cell(0, 5, f"  - {w.strip()}")

    matched = doctor[doctor['Disease'] == disease]
    if not matched.empty:
        specialist = matched['Specialized Doctor'].values[0]
        pdf.multi_cell(0, 5, f"Consultation: Consult a nearby {specialist}.")

    # Output PDF
    pdf_bytes = pdf.output(dest='S').encode('latin1')
    return io.BytesIO(pdf_bytes)


@app.post("/predict", summary="Predict disease from symptoms")
async def predict_and_generate(input_data: PatientInput):
    disease = get_predicted_value(input_data.symptoms)
    desc, pre, med, die, wrkout = helper(disease)

    report = {
        "name": input_data.name,
        "age": input_data.age,
        "gender": input_data.gender,
        "phone": input_data.phone,
        "email": input_data.email,
        "symptoms": input_data.symptoms,
        "disease": disease,
        "description": desc,
        "precautions": pre,
        "medications": med.split(', '),
        "diet": die.split(', '),
        "workouts": wrkout,
    }

    return JSONResponse(content=report)


@app.post("/download-pdf", summary="Download PDF report")
async def download_pdf(input_data: PatientInput):
    disease = get_predicted_value(input_data.symptoms)
    desc, pre, med, die, wrkout = helper(disease)
    pdf_buffer = generate_patient_pdf(input_data, disease, desc, med, die, wrkout)

    filename = f"{input_data.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_report.pdf"
    return FileResponse(pdf_buffer, media_type="application/pdf", filename=filename)


if __name__ == "__main__":
    uvicorn.run("main:main", host="0.0.0.0", port=8000, reload=True)
