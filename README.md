# 🩺 MedAI - Smart Disease Prediction & Health Report Generator

**MedAI** is an intelligent, user-friendly web application designed to assist users in identifying potential diseases based on symptoms using a trained ML model. It generates a detailed medical report including description, medications, diet plan, workout tips, and consultation guidance — all downloadable as a professional PDF.
<p align="center">
  <a href="https://medaiapp.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀 Try%20Live%20Demo-Streamlit%20App-brightgreen?style=for-the-badge" alt="Try MedAI" />
  </a>
</p>

---

## 🚀 Features

- ✅ Predicts disease from selected symptoms using a trained ML model.
- 📝 Generates a comprehensive medical report.
- 💊 Suggests medications and diet plans.
- 🏃 Recommends workouts for better recovery.
- 👨‍⚕️ Suggests specialist doctors based on the disease.
- 📄 Downloadable PDF report for future reference.

---


---

## 🧠 How It Works

1. **Input**: Users select symptoms from a predefined list.
2. **Prediction**: A pre-trained ML model predicts the most likely disease.
3. **Helper Function**: Gathers description, medications, diets, workouts, and specialist info.
4. **Output**: Information is displayed on the interface.
5. **PDF Report**: All info is compiled into a professional PDF report.

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Arup03NITRR/MedAI.git
cd MedAI
```
### 2. Install Dependencies
Ensure Python 3.7+ is installed, then:
```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
streamlit run app.py
```

## ✨ Future Improvements
- Add multi-disease probability ranking
- Enable user authentication
- Integration with health APIs
- Add chatbot-based interaction