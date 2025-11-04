<div align="center">
  <img src="output image.PNG" width="50%"/>

</div>
<h1>🏡 House Price Prediction</h1>

An end-to-end machine learning project to predict house prices using regression models. Built with Python and deployed via a Flask web app to demonstrate data science, model engineering, and full-stack skills.

<b>🔍 Project Overview:</b>
This project develops and deploys a House price prediction system using historical housing data. It involves:
- Data preprocessing & feature engineering  
- Model training, evaluation & comparison  
- Serialization of the model pipeline  
- Web interface to input house features and predict pricing  

It aims to showcase skills in machine learning, data engineering,  & full-stack deployment — making it ideal work sample material for roles in Data Science, Machine Learning Engineering, or Predictive Analytics.

<b>🛠️ Key Features & Components:</b>
- Data Handling & Feature Engineering 
  Clean, transform, and engineer features from raw housing data to improve model accuracy.
- Model Training & Evaluation
  Train multiple regression models (e.g. linear regression, tree-based) and compare performance metrics (RMSE, MAE, R²).
- Pipeline & Serialization
  Build a robust data + model pipeline and serialize it (e.g. using `pickle` or `joblib`) for reuse in production.
- Web Application (Flask)
  A user-friendly UI where users enter house attributes and get a price prediction in real time.

<b>📂 Project Structure:</b>

House-Price-Prediction/

├── app.py # Flask app entry point

├── House_price_prediction.ipynb # Notebook showing EDA, modeling, experiments

├── kc_house_data.csv # Raw housing dataset

├── house_price_pipeline.pkl # Serialized preprocessing + model pipeline(Model)

├── requirements.txt # Project dependencies

└── README.md # This readme


<b>💻 Technologies & Skills Demonstrated:</b>

| Area | Tools / Libraries |
|------|--------------------|
| Data Processing & EDA | pandas, numpy, seaborn, matplotlib |
| Machine Learning | scikit-learn, xgboost / random forest (if used) |
| Model Pipeline & Serialization | scikit-learn pipelines, joblib / pickle |
| Web Deployment | Flask |
| Version Control & Project Structure | Git & GitHub |

<b>🚀 Installation & Running:</b>
Prerequisites 
- Python 3.7+  
- pip (Python package installer)  

Steps
1. Clone the repo:  
   ```bash
   git clone https://github.com/UK183/House-Price-Prediction.git
   cd House-Price-Prediction
   
2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Flask application:
    ```bash
    python app.py

Open your browser and navigate to http://127.0.0.1:5000 to use the prediction UI.


<b>📈 Performance & Results:</b>

Model evaluation metrics (RMSE, MAE, R²) documented in the notebook.

Comparative performance across different algorithms shown with visualizations.

Deployed model used via API endpoint in Flask app.

You can see all results, experiments, and visualizations in House_price_prediction.ipynb.


<b>🧠 Key Learnings & Achievements:</b>

End-to-end experience—from raw data to live web app deployment.

Hands-on exposure to feature engineering, model tuning, and deployment challenges.


Ability to package ML pipelines for production and embed them in a user interface.



⚠️ Note: This project is for educational/demonstration purposes. Real-world implementation would require additional considerations like more extensive validation, error handling, security, and scalability.

---

### 👤 Author
[**Kazi Umar**](https://github.com/UK183)<br>
Linkedin profile: https://www.linkedin.com/in/umar-kazi18  
💼 Data Analyst | ML Engineer | Data Science & AI Enthusiast | Power BI | Python | SQL
