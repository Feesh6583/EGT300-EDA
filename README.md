1. Student Details

Name: Yu Jie & Zhao Ji

2. Project Overview

This repository contains an end-to-end Machine Learning Pipeline (MLP) designed to:
Load and process the bmarket_clean_final.db dataset
Perform feature preprocessing
Train and evaluate three ML models
Save trained models
Generate evaluation visualizations (confusion matrices & ROC curves)
This pipeline supports automated experiments and can be executed using one command.

3. Folder Structure
EDA Project/
├─ .git/
├─ .venv/
├─ data/
│  ├─ bmarket_clean_final.db
│  └─ processed/
│     ├─ X_train.csv
│     ├─ X_test.csv
│     ├─ y_train.csv
│     ├─ y_test.csv
│     └─ model_results.json
├─ models/
│  ├─ logistic_regression.joblib
│  ├─ random_forest.joblib
│  └─ grad_boost.joblib
├─ reports/
│  ├─ confusion_logistic_regression.png
│  ├─ confusion_random_forest.png
│  ├─ confusion_grad_boost.png
│  ├─ roc_logistic_regression.png
│  ├─ roc_random_forest.png
│  └─ roc_grad_boost.png
├─ src/
│  ├─ config.py
│  ├─ data_loader.py
│  ├─ models.py
│  ├─ pipeline.py
│  ├─ preprocessing.py
│  └─ train.py
├─ Dockerfile
├─ docker-compose.yml
├─ run.sh
├─ requirements.txt
├─ eda.ipynb
└─ README.md




4. How to Run the Pipeline

Option 1 — Using Virtual Environment
# Activate virtual environment
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate # Mac/Linux
# Run the pipeline
python -m src.pipeline

Option 2 — Using Docker
# Build Docker image
docker build -t eda_project .
# Run pipeline in container (mount local data folder)
docker run --rm -v ${PWD}/data:/app/data eda_project

This will automatically:
Load SQLite database
Preprocess the dataset
Split into train/test sets
Train all ML models
Save trained models
Generate ROC curves & confusion matrices
Produce a results summary

5. Dataset

The dataset used is:
data/bmarket_clean_final.db

Contains client attributes, marketing campaign results, contact history, and demographic details

Processed training/testing files are stored in: data/processed/

6. Pipeline Workflow

The pipeline follows this sequence:
1. Load Dataset
Reads from SQLite using sqlite3 into pandas.
2. Preprocessing
Performed in preprocessing.py:
Handle missing values
Encode categorical variables
Scale numerical fields
Split dataset (train/test)
Save .npy files
3. Train 3 ML Models
Located in train.py:
Logistic Regression
Random Forest
Gradient Boosting
Each model outputs:
Accuracy
Precision
Recall
F1-score
ROC AUC
4. Generate Visualizations
For each model:
Confusion Matrix
ROC Curve
Saved in /reports.
5. Save Models
All models are stored as .joblib files:
models/


7. Model Performance Summary (From Actual Output)

Model	            Accuracy	Precision	Recall	F1-score	ROC-AUC
Logistic Regression	0.6953  	0.2068	    0.6013	0.3078	    0.7191
Random Forest	    0.8362	    0.2714	    0.2694	0.2704	    0.6544
Gradient Boosting	0.8950  	0.6019	    0.2004	0.3007	    0.7220

Best Overall Model: Gradient Boosting
Chosen because:
Highest accuracy (0.8950)
Highest ROC-AUC (0.7220, best discriminator)
Balanced performance given the imbalanced dataset


8. Feature Processing Summary
Feature Type	        Processing Applied
Numerical features   	StandardScaler()
Categorical features	OneHotEncoder()
Missing values	        Filled / encoded appropriately
Target column	        Label encoded (0/1)
Dataset split	        80% train / 20% test

All transformations are done programmatically in preprocessing.py.


9. Reason for Model Choices

Logistic Regression
Strong baseline classifier
Easy to interpret
Useful for initial comparison

Random Forest
Handles complex interactions
Robust to noise
Suitable for tabular datasets

Gradient Boosting
Excels in structured data tasks
Often achieves highest predictive power
Handles class imbalance better
This model achieved the best performance, making it the final recommended model.

10. Evaluation Metrics & Why They Are Used
Metric	                Purpose
Accuracy	            Overall correctness
Precision	            How many predicted positive cases were correct
Recall (Sensitivity)	Ability to capture true positives (important in banking campaigns)
F1-score	            Balance between precision & recall
ROC AUC	                Measures ability to separate positive vs negative classes

Since banking subscription data is imbalanced, ROC-AUC and Recall are especially important.

11. Deployment Considerations

To deploy this pipeline in production:
Export only the best model (grad_boost.joblib)
Ensure preprocessing pipeline is identical
Serve via REST API / batch scoring system
Monitor for data drift
Re-train model when new campaign data is available

12. Conclusion

This Machine Learning Pipeline:

Fully automates data processing
Trains three competitive ML models
Saves trained models
Provides robust evaluation visualizations
Identifies Gradient Boosting as the best-performing classifier
Supports Docker-based reproducibility
