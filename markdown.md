<div align="center" style="border: 2px solid #ccc; padding: 20px; border-radius: 12px; width: 80%; margin: auto; box-shadow: 0 0 10px rgba(0,0,0,0.15);">
    <img
        width="180"
        height="220"
        alt="Logo - SURE ProEd"
        src="https://github.com/user-attachments/assets/88fa5098-24b1-4ece-87df-95eb920ea721"
        style="border-radius: 10px;"
    />

  <h1 align="center" style="font-family: Arial; font-weight: 600; margin-top: 15px;">SURE ProEd (formerly SURE Trust) 
      </h1>
<h2 style="color: #2b6cb0; font-family: Arial;">Skill Upgradation for Rural youth Empowerment Trust</h2>
</div>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<div style="padding: 20px; border: 2px solid #ddd; border-radius: 12px; width: 90%; margin: auto; background: #fafafa; font-family: Arial;">

<h2 style = "color:#333;"> Student Details </h2>
<div align = "left" style ="margin: 20px; font-size: 16px;">
  <p><strong>Name:</strong> Mahajan Ashok</p>
  <p><strong>Email ID:</strong> asokroshant78@gmail.com</p>
  <p><strong>College Name:</strong> TKR College of Engineering and Technology</p>
  <p><strong>Branch/Specialization :</strong> B.Tech - Computer Science (Data Science)</p>
  <p><strong>College ID:</strong> 22K91A6793</p>
</div>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<h2 style="color:#333;"> Course Details </h2>
<div align="left" style="margin: 20px; font-size: 16px;">
  <p><strong>Course Opted:</strong> Data Science with Python</p>
  <p><strong>Instructor Name:</strong> Mr. Gaurav Patel</p>
</div>
<div align="left" style="margin: 20px; font-size: 16px;">
  <p><strong>Duration:</strong> July 2025 - March 2026</p>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

<h2 style="color:#333;"> Trainer Details </h2>
<div align="left" style="margin: 20px; font-size: 16px;">

<p><strong>Trainer Name:</strong> Gaurav Patel</p>
<p><strong>Trainer Email ID:</strong> gaurav.patel.gpp@gmail.com</p>
<p><strong>Trainer Designation:</strong> Tutor</p>

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

## **Table of Contents**

- [Course Learning](#course-learning-to-be-edited-by-student)
- [Projects Completed](#projects-completed)
- [Project Introduction](#project-introduction)
- [Technologies Used](#technologies-used)
- [Roles and Responsibilities](#roles-and-responsibilities)
- [Project Report](#project-report)
- [Learnings from LST & SST](#learnings-from-lst--sst)
- [Community Services](#community-services)
- [Certificate](#certificate)
- [Acknowledgments](#acknowledgments)

<hr style="border: 0; border-top: 1px solid #ccc; width: 80%;" />

## Overall Learning

During this internship, I worked on building a complete AI-based project from idea to implementation. I learned how to connect machine learning concepts with practical development by working on backend logic, frontend integration, and overall system flow. This experience improved my problem-solving, debugging, and documentation skills. It also helped me grow in communication, time management, and professional discipline while working in a structured environment.

<h2 style="color:#333;"> Projects Completed </h2>
<div align="left" style="margin: 20px; font-size: 16px;">

<p><strong><a href="#project1">Project 1:</a></strong> Ensemble Learning for Sensor-Based Anomaly Detection in Energy Manufacturing Systems</p>

</div>

<!-- Project 1 -->
<h3 id="project1">Project 1: Ensemble Learning for Sensor-Based Anomaly Detection in Energy Manufacturing Systems</h3>
<p>
  This project focuses on detecting abnormal patterns in sensor data from energy manufacturing systems using ensemble learning methods. It combines data preprocessing, model training, and evaluation to improve reliability and early fault detection in industrial operations.
</p>
<p>
  <strong>→ Full Project Report: Completed</strong>
</p>

## Project Introduction

This project was developed to identify anomalies early in energy manufacturing systems using sensor-based data. Since the dataset is highly imbalanced, we focused on building a robust ensemble pipeline that can improve anomaly detection performance and reduce missed critical events.

## Technologies Used

- Python
- Pandas, NumPy
- LightGBM, XGBoost
- Scikit-learn (Stratified K-Fold, evaluation metrics)
- Streamlit (frontend dashboard)
- Joblib, JSON

## Roles and Responsibilities

- Performed data preprocessing and feature engineering from sensor and date signals.
- Trained LightGBM and XGBoost models using 5-fold stratified cross-validation.
- Tuned thresholds using F1-based selection and evaluated with AUC-ROC and AUC-PR.
- Built an ensemble strategy and exported model artifacts.
- Developed a Streamlit frontend for batch prediction and model-wise inference.

## Project Report

### What We Did (Backend)

The backend is implemented in `pipeline.py`. We loaded `train.parquet` and `test.parquet`, engineered domain-informed features (date-based features, decoded transformations, and interaction terms), and trained both LightGBM and XGBoost with stratified 5-fold cross-validation to handle class imbalance. We then evaluated each model, selected thresholds, created ensemble predictions, and saved artifacts in the `models/` folder (`lgbm_fold_*.pkl`, `xgbm_fold_*.pkl`, `feature_cols.pkl`, `eval_summary.json`) along with `submissions.csv`.

### Frontend Usage

The frontend is implemented in `app.py` using Streamlit. It loads trained artifacts and provides a UI for anomaly prediction.

1. Run backend once to generate model artifacts:

- `python pipeline.py`

2. Start frontend app:

- `streamlit run app.py`

3. In the app sidebar, choose model option:

- `Auto (best AUC-PR)` or `Ensemble` or `LightGBM` or `XGBoost`

4. Upload input file (`.csv` or `.parquet`) with required columns:

- `Date, X1, X2, X3, X4, X5`

5. View output:

- anomaly probability,
- predicted label based on active threshold,
- model summary and distribution views.

### Result

We completed an end-to-end anomaly detection system with a trained backend ML pipeline and a usable frontend interface for real-time/batch-style file-based predictions.

<hr style="height:1px; border-top:1px solid #ccc; width:80%;" />

## **References**

- [Wikipedia](https://wikipedia.com)
<!--you can add refrences over here in same syntax as above -->

---

## **Learnings from LST and SST**

<!-- add your experiences over here -->

LST and SST sessions helped me improve both personal and professional skills. I learned how to communicate clearly, work better in teams, manage time effectively, and stay confident during presentations and discussions. These sessions also taught me discipline, workplace behavior, and a positive approach to solving problems, which supported my technical learning throughout the internship.

## LST and SST sessions helped me....

## **Community Services**

<!-- add descreption in your own words -->

During my internship period, I participated in multiple community-oriented activities .....<!-- add descreption in your own words -->

### **Activities Involved**

<!-- add the location where you given -->

- **Blood Donation** – Donated blood and supported basic assistance tasks during the camp.

 <!-- add the location where you have panted -->

- **Tree Plantation Drive** – Participated by planting trees and contributing to environmental improvement.

  <!-- add the location where you helped -->

- **Helping Elder Citizens** – Assisted two elderly individuals with simple daily tasks and provided support where needed.

<!-- you can write impacts according to your experience in your words-->

### **Impact / Contribution**

- Helped create a supportive environment during the blood donation camp. <!-- add the location where you given -->
- Actively participated in promoting a greener and cleaner surroundings.
- Offered personal assistance to elder citizens, strengthening community bonds.
- Improved skills in communication, coordination, and social responsibility.

### **Photos**

<!-- add your photos below -->
<!-- change url below with your image urls (inside  src='')-->

- These are just placeholder (sample) images <!-- remove this line -->

<div align="center">
<img src="https://media.licdn.com/dms/image/v2/D561FAQEJNBia4UCa5w/feedshare-document-images_800/B56Zm5b6SJJkAg-/1/1759752731458?e=1766016000&v=beta&t=7GABy91-0FNbir386wPdJ-Grr385JzS3tR5LQIw1CWg" alt="Community Service Photo 1" width="30%">
<img src="https://media.licdn.com/dms/image/v2/D561FAQEJNBia4UCa5w/feedshare-document-images_800/B56Zm5b6SJJkAg-/2/1759752731458?e=1766016000&v=beta&t=6RfJQWWqlQUPcCvDnQNW7kR6yf7w-wPDsIPxum409ck" alt="Community Service Photo 2" width="30%">
<img src="https://media.licdn.com/dms/image/v2/D561FAQEJNBia4UCa5w/feedshare-document-images_800/B56Zm5b6SJJkAg-/3/1759752731458?e=1766016000&v=beta&t=yWaunKdRdLUKBLbmM3UjRYYz-_GSCfWEQ3_R7dW0xLM" alt="Community Service Photo 3" width="30%">
</div>

---

## **Certificate**

The internship certificate serves as an official acknowledgment of the successful completion of my training period. It will be issued by the organization upon fulfilling all required tasks and meeting the performance expectations of the program. The certificate validates the skills, experience, and contributions made during the internship.

<!-- add your certificate image url below (inside src='')-->

<p align="center">
<img src="https://github.com/Lord-Rahul/Practice-Programs/blob/main/react/1/public/Gemini_Generated_Image_a6w8rda6w8rda6w8.png?raw=true" alt="Internship Certificate" width="80%">
</p>

---

## **Acknowledgments**

<!-- you can add Acknowledgments over here in same syntax as below . eg trainer name , company name , role etc -->

- [Prof. Radhakumari Challa](https://www.linkedin.com/in/prof-radhakumari-challa-a3850219b) , Executive Director and Founder - [SURE Trust](https://www.suretrustforruralyouth.com/)
