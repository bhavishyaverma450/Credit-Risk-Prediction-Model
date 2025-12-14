💳 Credit Risk Prediction using Machine Learning

An end-to-end machine learning project that predicts loan approval risk by comparing multiple regression and classification models on a real-world credit dataset.

📌 Overview

This project focuses on building a complete ML pipeline — from data preprocessing and exploratory analysis to model training, evaluation, and comparison — to identify the most reliable model for credit risk assessment.

⚙️ Features

Data cleaning and preprocessing

Categorical encoding and feature scaling

Exploratory Data Analysis (EDA)

Multiple ML model implementations

Model performance comparison using metrics and plots

Model saving using Pickle

🤖 Models Used
Regression

Simple Linear Regression

Multiple Linear Regression

Polynomial Regression

Classification

Logistic Regression

Decision Tree

Random Forest

K-Nearest Neighbors (KNN)

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Model performance is visualized using accuracy comparison bar charts and Precision–Recall–F1 plots.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

🧠 Key Insights

Random Forest delivered the best overall performance

Feature scaling improved Logistic Regression and KNN

Evaluating multiple metrics provides better model reliability than accuracy alone

🚀 How to Run
pip install -r requirements.txt
python main.py

📁 Project Outputs

model.pkl – trained Random Forest model

scaler.pkl – fitted StandardScaler

🔮 Future Work

Hyperparameter tuning

ROC-AUC analysis

Model deployment using Flask / FastAPI

⭐ If you find this project useful, consider starring the repository!
