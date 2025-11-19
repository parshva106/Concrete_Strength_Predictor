# 🏗️ **Concrete Strength Predictor**

*A Machine Learning Web Application for Predicting Concrete Compressive Strength*

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost)
![Machine Learning](https://img.shields.io/badge/ML-Regression-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 **Overview**

This project is a powerful and interactive **machine learning–based web application** that predicts the **compressive strength of concrete** (in MPa) using its material composition and age.
The model is built using **XGBoost Regression**, trained on the widely used **Concrete Compressive Strength Dataset**.

The entire application is deployed using **Streamlit**, offering:

* Real-time prediction
* Batch predictions via CSV
* Feature importance visualization
* Interactive plots
* Dedicated model analysis dashboard

---

## 🚀 **Features**

### 🔍 **1. Single Prediction Mode**

Adjust the material parameters using sliders to get an instant prediction of concrete compressive strength.

### 📊 **2. Batch Prediction Mode**

Upload a CSV containing multiple samples and obtain predictions for all records.

### 🔬 **3. Model Analysis**

Explore:

* Feature importance charts
* Model behavior
* Feature ranges & descriptions

### ℹ️ **4. About Page**

Details about dataset, model, assumptions, and usage guidelines.

---

## 🧠 **Machine Learning Model**

* **Algorithm**: XGBoost Regressor

* **Task**: Regression

* **Features Used (8):**

  * Cement
  * Slag
  * Fly Ash
  * Water
  * Superplasticizer
  * Coarse Aggregate
  * Fine Aggregate
  * Age

* **Target Variable**: Concrete Compressive Strength (MPa)

The model demonstrates **high predictive accuracy** and is suitable for experimental, research, and educational use.

---

## 📂 **Project Structure**

```
Concrete_Strength_Predictor/
│
├── app.py                     # Streamlit Web App
├── model.pkl                  # Trained XGBoost Model
├── Concrete_Data_Yeh.csv      # Dataset used for training
├── README.md                  # Project Documentation
└── requirements.txt           # Required Python libraries
```

---

## 🛠️ **Tech Stack**

| Component        | Technology    |
| ---------------- | ------------- |
| Frontend UI      | Streamlit     |
| ML Model         | XGBoost       |
| Data Handling    | Pandas, NumPy |
| Visualizations   | Plotly        |
| Backend Language | Python        |

---

## 📥 **Installation Guide**

### **1. Clone the repository**

```bash
git clone https://github.com/your-username/Concrete_Strength_Predictor.git
cd Concrete_Strength_Predictor
```

### **2. Install dependencies**

```
pip install -r requirements.txt
```

### **3. Run the Streamlit app**

```
streamlit run app.py
```

---

## 🧪 **How to Use**

### ✔️ **Single Prediction**

* Move the sliders to set ingredient quantities
* Click on **“Predict Concrete Strength”**
* View:

  * MPa prediction
  * Strength Category
  * Feature Impact

### ✔️ **Batch Prediction**

* Upload a CSV containing the 8 required features
* Get:

  * Predictions for all samples
  * Downloadable CSV
  * Strength distribution plot

### ✔️ **Model Analysis**

* Pie chart of feature contributions
* Summary of model details
* Feature ranges & insights

---

## 📈 **Visualizations Included**

* Feature importance bar chart
* Feature contribution pie chart
* Strength distribution histogram
* Metrics panels for summary insights

All charts are implemented using **Plotly** for a smooth UI experience.

---

## 📊 **Dataset Reference**

The training dataset is based on:

> Yeh, I-Cheng (1998). **Concrete Compressive Strength Data Set.**

Contains **1030** samples of concrete mixes and their measured compressive strength.

---

## ⚠️ **Important Disclaimer**

This ML application provides predictions based on experimental data and should only be used as a **supplementary tool**.
Always validate with **professional engineering expertise** and **laboratory tests** before real-world use.

---

## 📝 **License**

This project is licensed under the **MIT License**.

---
