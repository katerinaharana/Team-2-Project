#!/usr/bin/env python
# coding: utf-8

# # *Deliverable 2*: Preprocessing
# 
# * Import dataset and necessary libraries
# <br><br>  
# * Handling missing values
#   * Drop missing mpg feature values
#   * Scaling data
#   * Fill hp missing values creating a custom function
#   <br><br>  
# * Handling outliers
#   * Detect Multivariate outliers using Isolation Forest
#   * Visualise outliers with 2D PCA
#   * Drop outliers 
#  <br><br>
# * Encoding categorical features
# <br><br>   
# 
# 

# ## Import dataset and libraries

# In[22]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install -U nbformat==4.2.0')


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import os
nb_path = os.path.abspath('')
file_path = os.path.join(nb_path, '../../data/df_eda.csv')
df = pd.read_csv(file_path)


# In[23]:


df


# ## Handling missing values & scaling the data 

# In[24]:


missing_values = df.isnull().sum()
missing_values


# Missing values for mpg, horsepower and model.
# Since mpg is the target column, it is better not to be filled with a predicted value.
# Missing model value is not a concern in regards to our analysis. 
# Horsepower will be filled using a custom function for a more educated guess on predicted value

# ### Dropping mpg N/As

# In[25]:


# Drop rows where 'mpg' is missing
df = df.dropna(subset=['mpg'])


# ### Custom function for filling hp N/As <br>
# 

# For each car with a missing hp value find a similar car and use its hp value. To find the similar car, first scale the data, then for every missing hp car, substract its feature values from all other cars' respective feature values. keep the absolute value of the substraction (distance) and for each car create a score by suming all feature distances. The car with the lowest score is the most similar car to the one in question. Fill the missing hp value with the one of the most similar car  

# In[26]:


# Scale features 
scaler = StandardScaler()
features_to_scale = ['mpg','cylinders', 'displayments', 'horsepower','weight', 'acceleration']
df_scaled = df.copy()
df_scaled[features_to_scale] = scaler.fit_transform(df[features_to_scale])
df_scaled.head()
df_scaled.loc[[38, 75]]


# In[8]:


# Define the function to find the most similar car and impute missing values
def similar_car(target_col, features):
    """
    Impute missing values in the target column by finding the most similar car.
    
    Parameters:
        target_col (str): The name of the column to predict 
        features (list): List of features to use for similarity comparison.
    """
    imputed_values = []  # To store the imputed values and their sources
    
    for idx, row in df_scaled[df_scaled[target_col].isnull()].iterrows():
        # Exclude the row with missing value
        others = df_scaled[df_scaled[target_col].notnull()]
        
        # Compute absolute differences and sum them row-wise 
        differences = others[features].sub(row[features], axis=1).abs().sum(axis=1)
        
        # Find the index of the most similar car (minimum difference)
        most_similar_idx = differences.idxmin()
        
        # Assign the target column value from the most similar car
        imputed_value = df_scaled.loc[most_similar_idx, target_col]
        df_scaled.loc[idx, target_col] = imputed_value
        
        # Store the imputed value and corresponding similar car
        imputed_values.append((idx, imputed_value, most_similar_idx))
    
    # Display the imputed values
    print("Imputed Values for Missing Entries in '{}':".format(target_col))
    for idx, value, similar_idx in imputed_values:
        print(f"Row {idx}: Imputed Value = {value} (Based on Similar Car at Row {similar_idx})")

# Perform similar car imputation for 'horsepower' 
features = ['mpg','cylinders', 'displayments', 'weight', 'acceleration']
similar_car(target_col='horsepower', features=features)

# Check the updated 'horsepower' column
df_scaled[['horsepower']]
df_scaled.head()


# In[9]:


df_scaled.isnull().sum()


# ## Handling outliers
# As noticed in the 1st Deliverable there are some outliers considering the range of each feature, although we should consider multivariate outliers

# ### Isolation Forest: Multivariate outlier detection(based on the combination of all other features)
# 
# A value might not be an outlier in its own column but could be an outlier in combination with other features. <br>
# 
# 
# 

# In[10]:


# Define columns to use for outlier detection
columns = ['mpg', 'displayments', 'weight', 'cylinders', 'horsepower', 'acceleration']

# Fit Isolation Forest
iso_forest = IsolationForest(n_estimators=600, contamination=0.05
                             , random_state=42)
df_scaled["anomaly_score"] = iso_forest.fit_predict(df_scaled[columns])

# Mark outliers
df_scaled["multivariate_outlier"] = df_scaled["anomaly_score"].map({1: "Normal", -1: "multivariate_outlier"})

# Count outliers
outlier_count = df_scaled["multivariate_outlier"].value_counts()
print("multivariate outlier count:\n", outlier_count)

# Interactive 3D Scatter Plot (Coloring Outliers in Green)
fig = px.scatter_3d(
    df_scaled,
    x="weight",
    y="displayments",
    z="mpg",
    color="multivariate_outlier",  # Highlighting outliers
    color_discrete_map={"Multivariate_outlier": "green", "Normal": "blue"},
    title="Isolation Forest Outlier Detection for Selected Features",
    labels={"weight": "Weight", "displayments": "Displacement", "mpg": "Miles Per Gallon"}
)
# Adjust the size of the plot
fig.update_layout(
    width=1000,  # Set the width in pixels
    height=800   # Set the height in pixels
)
fig.show()


# ### Visualise mutlivariate outliers by projecting dataset in a 2dimensional space using PCA

# In[29]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_scaled[columns])
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=df_scaled['multivariate_outlier'])


# In[28]:


df_scaled


# In[30]:


# Get indices of multivariate outliers and compare them to feature outliers
multivariate_outliers = df_scaled[df_scaled['anomaly_score'] == -1].index.tolist()
outliers=df_scaled[df_scaled['outlier'] == "Outlier"].index.tolist()
all_outliers = list(set(multivariate_outliers + outliers))

# Print the number of outliers found
print(f"Total Multivariate Outliers: {len(multivariate_outliers)}")
print(f"Total Feature Outliers: {len(outliers)}")
print(f"Total Unique Outliers to Remove: {len(all_outliers)}")

# Print or use the list of outlier indices if needed
print("Indices of multivariate outliers:", multivariate_outliers)
print ("Indices of outliers:", outliers)


# ### Drop multivariate & feature outliers

# In[14]:


# Drop the combined outliers from the dataset
df_scaled_cleaned = df_scaled.drop(index=all_outliers).reset_index(drop=True)

# Print new dataset shape
print(f"Original Dataset Shape: {df_scaled.shape}")
print(f"Cleaned Dataset Shape: {df_scaled_cleaned.shape}")


# ## Encoding categorical features

# In[15]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

# label encoding to the 'car brand' column
df_scaled_cleaned['car_name_encoded'] = label_encoder.fit_transform(df_scaled_cleaned['car brand'])

print(df_scaled_cleaned[['car brand', 'car_name_encoded']])


# In[16]:


# Display the mapping of numbers to car brands
mapping = dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))
print("Label Encoding Mapping:")
for brand, number in mapping.items():
    print(f"{number}: {brand}")


# Note: we now have 28 car brands instead of 30 since one was dropped as an outlier (hi) and an other as a missing mpg value(citroen)

# In[17]:


#cleand but not scaled dataset
# Define the features that were scaled
features_to_unscale = ['mpg', 'cylinders', 'displayments','horsepower' 'weight', 'acceleration', 'model year']

# Inverse transform to restore original values
df_cleaned = df_scaled_cleaned.copy()
df_cleaned[features_to_scale] = scaler.inverse_transform(df_scaled_cleaned[features_to_scale])


# In[18]:


df_cleaned


# In[19]:


#sorting by mpg values
df_cleaned= df_cleaned.sort_values(by='mpg')
 
#calculating 33.3rd and 66.6th percentiles
q1 = df_cleaned['mpg'].quantile(0.333)
q2 = df_cleaned['mpg'].quantile(0.666)
df_cleaned['mpg_classes'] = pd.qcut(df_cleaned['mpg'], q=[0, 0.333, 0.666, 1], labels=[0, 1, 2])
 
#distribution
distribution = df_cleaned['mpg_classes'].value_counts()
 
#results
print("Quantile thresholds:")
print(f"q1: {q1}")
print(f"q2: {q2}")
print("\nLabel distribution:")
print(distribution)


# In[20]:


# Save the reverted dataset
df_cleaned.to_csv(os.path.join(nb_path, '../../data/df_cleaned.csv'), index=False)


# In[21]:


get_ipython().system('jupyter nbconvert --to script D2.ipynb')

