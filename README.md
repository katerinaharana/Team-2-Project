# Task Overview

This project is part of the WE LEAD Bootcamp on Data Science & Business Intelligence. The objective is to develop a classification model that predicts a car's fuel efficiency category based on various attributes. The goal is to create a proof-of-concept tool that could help car rental companies make data-driven decisions when updating their fleets.

**Team Members** (alphabetical order):

Alexaki Erofili

Diamanti Ioli

Karagianni Ioanna

Charana Aikaterini

# Dataset 

We are using the  mpg.data Dataset, which consists of 406 instances with the following 9 attributes:

- MPG
- Cylinders
- Displacement
- Horsepower
- Weight
- Acceleration
- Model Year
- Origin 
- Car Name

# Deliverables

**D1:** Exloratory Data Analysis (EDA) (https://github.com/katerinaharana/Team-2-Project/blob/main/notebooks/deliverable-1/D1.ipynb)

**D2:** Data Preprocessing (https://github.com/katerinaharana/Team-2-Project/blob/main/notebooks/deliverable-2/D2.ipynb)

**D3:** Models and Evaluation (https://github.com/katerinaharana/Team-2-Project/tree/main/notebooks/deliverable-3)

## D1: Exploratory Data Analysis (EDA)
We conducted an in-depth analysis to understand data structure, detect anomalies, and explore relationships between variables:

Missing Values & Duplicates: Detected and documented, including 8 missing mpg and 6 missing horsepower.

Feature Distribution: Visualized with histograms.

Correlation Analysis: Strong negative monotonic relationships found between mpg and displacement, horsepower, and weight. Spearman and Pearson coefficients supported these findings.

Outlier Detection:

Used boxplots and scatterplots for 1D outliers.

IQR method identified 15 significant anomalies.

Categorical Feature Engineering:

Extracted car brand from Car Name.

Visualized average mpg by brand and origin.

Explored how consumption trends evolved across model years.

## D2: Data Preprocessing
To prepare the dataset:

Handling Missing Values:

Dropped rows with missing mpg.

Filled missing horsepower via a custom similarity-based function that compared scaled distances between cars.

Outlier Removal:

Detected multivariate outliers using Isolation Forest.

Combined univariate and multivariate approaches to identify and drop 23 outliers.

Feature Engineering & Encoding:

Engineered car brand as a feature.

Encoded categorical variables numerically.

Label Creation:

Experimented with cluster-based and PCA-guided binning.

Final classification used quantile-based binning, producing three balanced categories (Low, Medium, High fuel efficiency).

🤖 D3: Modeling & Evaluation
We trained and evaluated multiple classification models:

## Ensemble Methods
Models Used:

AdaBoost

Gradient Boosting

XGBoost

Random Forest

Bagging Classifier (SVM base)

Stacking Classifier (Random Forest + XGBoost + KNN)

Techniques:

Train-test split (70-30)

GridSearch for hyperparameter tuning

Cross-validation (k-fold)

Metrics: Accuracy, F1-score

Results:

Gradient Boosting performed best with ~81% accuracy and F1.

Stacking and Random Forest were consistent but slightly lower.

Boosting methods proved superior in reducing overfitting and increasing generalization.

## Neural Networks
Implemented a deep feedforward neural net:

Architecture: 4 hidden layers (64 → 32 → 16 → output)

Activation: tanh

Loss: Cross-entropy

Optimizer: Adam with ReduceLROnPlateau

Validation: Stratified k-fold

Regularization: Early stopping to avoid overfitting

Results:

Achieved accuracy and F1-score ~98%, outperforming ensemble methods significantly.

# Final Model Evaluation scores 

Histogram with the different model performance scores (Accuracy and F1 Macro) can be found here: https://github.com/katerinaharana/Team-2-Project/blob/main/notebooks/deliverable-3/Model_Performance_Comparison.ipynb

The different scores of the different models are as seen on this table: 

<img width="669" alt="Screenshot 2025-02-25 at 20 55 28" src="https://github.com/user-attachments/assets/ac8f1df8-46bb-4a78-b26d-811991399991" />



# Setup

## Create Virtual Enviroment 
*Please ensure you have installed python 3.11*

### MacOS
In terminal:

```python3.11 -m venv venv```

```source venv/bin/activate```

```pip install -r requirements.txt```

Run notebooks in deliverable-1, deliberable-2 and the notebooks in deliverable-3 (D3_Enseble_Models.ipynb, D3_NN.ipynb, D3_knn.ipynb) in that suggested order.

### Windows 
In terminal: 

```python3.11 venv venv```

```venv\Scripts\Activate```

If security error arises try running: ```Set-ExecutionPolicy Unrestricted -Scope Process```

Then again: ```venv\Scripts\Activate```

```pip install -r requirements.txt```

Run notebooks in deliverable-1, deliberable-2 and the notebooks in deliverable-3 (D3_Enseble_Models.ipynb, D3_NN.ipynb, D3_knn.ipynb) in that suggested order.
