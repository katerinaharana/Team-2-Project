#!/usr/bin/env python
# coding: utf-8

# # *Deliverable 1*: EDA (Exploratory Data Analysis)
# 
# * Import dataset and necessary libraries
# <br><br>  
# * Overview & Basic information about the dataset
#   * Deleting unneeded columns
#   * Detecting N/As & duplicates
#   * Descriptive statistics
#   * Range and value counts for each column
#   * Visualisation histograms (values in each column and counts of instances)
# <br><br>  
# * Relationship between the columns
#    * Scatterplots
#    * Spearman
#    * Pearson
#    * Interactive 3d plot
# <br><br>
# * Outliers
#    * Scatterplots
#    * Boxplots
#    * IQR method
# <br><br>   
# * Categorical analysis for brands
#   * Visualisation of counts of car brands
#   * Origin
# 
# 

# ## Import dataset and necessary libraries

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# In[4]:


file_path = 'C:/Users/Katerina/Team2 Project/Team-2-Project/data/mpg.data.csv'
df = pd.read_csv(file_path)


# ## Overview & Basic info about the dataset
# 

# In[6]:


df


# In[7]:


df.info()


# - The dataset includes 9 input features, namely: mpg, cylinders, displayments, horsepower, weight, acceleration, model year, origin, car name and it is composed of 405  samples.
# 
# - The features "mpg", "displayments", "horsepower" and "acceleration" are 64-bit floating point numbers, while the "cylinders", "weight", "model year" and "origin" are 64-bit integer numbers and the "car name" is object.
# 

# ### Delete unneeded columns

# In[10]:


df = df.drop(columns=['Unnamed: 9', 'Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12'])


# In[11]:


df.head()


# In[12]:


df.nunique(axis=0)


# ### Detect Missing Values & Duplicates

# In[14]:


missing_values = df.isnull().sum()
missing_values


# - The feature "mpg" has 8 missing data and the feature "horsepower" has 6 missing data.

# In[16]:


# find total duplicate entries and drop them if any
print(f'total duplicate rows: {df.duplicated().sum()}')

# drop duplicate rows if any
df = df[~df.duplicated()]
df.shape


# ### Descriptive statistics

# In[18]:


df.describe()


# ### Range & count of values of the different columns in the dataset

# In[20]:


# print the range of the features
for i in df.columns[:8]:
    print(f"The range of the feature'{i}' is [{df[i].min()},{df[i].max()}]")


# In[21]:


# dic for percentages
percentages_dict = {}

# Select numerical columns
numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns

for col in numerical_columns:
    # unique value counts
    print(f"Value counts for {col}:")

    # columns with decimals
    if col == 'mpg' or col == 'acceleration':  # Corrected condition
        df[col] = df[col].round(1)  # Round mpg and acceleration values to 1 decimal place

    # print value counts
    value_counts = df[col].value_counts()
    print(value_counts)

    # printing the most frequent value and its count
    most_frequent_value = value_counts.idxmax()
    most_frequent_count = value_counts.max()
    print(f"The most frequent value in {col} is {most_frequent_value} with {most_frequent_count} occurrences.")

    # store percentages for the column
    percentages = ((value_counts / value_counts.sum()) * 100).round(2)
    percentages_dict[col] = percentages

    print("\n" + "-"*40 + "\n")


# In[22]:


# percentages for each met value
for col, percentages in percentages_dict.items():
    print(f"Percentage of each value in {col}:")

    percentages = percentages.apply(lambda x: f"{x}%")

    print(percentages)
    print("\n" + "-"*40 + "\n")


# Here we wanted to show the percentages of each seperate value count met on the dataset.
# Our focus fell on cylinders and origin where there is a clear dominance of one value
# 
# **Cylinders:**
# 
# 4:    50.99%
# 
# 8:     26.6%
# 
# 6:    20.69%
# 
# 3:     0.99%
# 
# 5:     0.74%
# 
# **Origin:**
# 
# 1:    62.56%
# 
# 3:    19.46%
# 
# 2:    17.98%

# ### Visualisation histograms
# x axis: values of instances
# 
# 
# y axis: counts of each value

# In[25]:


sns.set_theme(style="whitegrid")
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns
plt.figure(figsize=(16, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df[col], bins=20, color='blue')
    plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()


# ## Relationship between the columns

# ### Scatterplots

# In[28]:


#MPG Trends Over the Years

df.groupby("model year")["mpg"].mean().plot(marker="o", figsize=(8, 5))
plt.title("Average MPG Over the Years")
plt.xlabel("Model Year")
plt.ylabel("MPG")
plt.show()


# In[29]:


# columns
columns = ['horsepower', 'weight', 'cylinders', 'acceleration', 'displayments','origin']

# scatter plots between mpg and columns
for col in columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='mpg', y=col)
    plt.title(f'mpg vs {col}')
    plt.xlabel('mpg')
    plt.ylabel(col)
    plt.show


# - The relationships between "mpg" with "displayments", "horsepower", "weight", are strong negative monotonic.
# - Between the "mpg" and "acceleration" there seems to be a weak positive monotonic trend, although the scatter plot shows significant variability.

# ### Spearman Correlation
# We use this type of correlation because our data (mpg and displacements, weight, horsepower) are not linear
# + Monotonic distributions
# + Robust to outliers
# + Measures the direction - the strength of a relationship

# In[32]:


# spearman correlation
spearman_correlation = df[numerical_columns].corr(method='spearman')
plt.figure(figsize=(10, 8))
sns.heatmap(spearman_correlation, annot=True, cmap='coolwarm', center=0, linewidths=0.5, fmt='.2f')
plt.title('Spearman Correlation Heatmap')
plt.show()


# - After performing Spearman correlation it is obvious that the greatest correlation of "mpg" is met upon: "weight", "displayments", "horsepower", "cylinders".

# ### Pearson heatmap

# In[35]:


plt.figure(figsize=(10, 8))
correlation_matrix = df[numerical_columns].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap")
plt.show()


# - From the pearson heatmap it is also evident that "cylinders", "displayments", "horse power" and "weight" are the features that mostly affect the "mpg".

# ### Interactive plot for the 2 most highly correlated to mpg features

# In[38]:


# Create Interactive 3D Scatter Plot
fig = px.scatter_3d(
    df,
    x="weight",
    y="displayments",
    z="mpg",
    color="mpg",  # Color by MPG
    title="Interactive 3D Scatter Plot: MPG vs Weight vs Displacement",
    labels={"Weight": "Weight", "Displacement": "Displacement", "MPG": "Miles Per Gallon"}
)

# Adjust the size of the plot
fig.update_layout(
    width=1000,  # Set the width in pixels
    height=800   # Set the height in pixels
)

# Show the Plot
fig.show()


# Lightweight Cars with Small Engines (left side, lower displacement, red points):
# These cars tend to have high MPG values (fuel-efficient).
# 
# Heavyweight Cars with Large Engines (right side, higher displacement, blue points):
# These cars are less fuel-efficient, as seen by their low MPG.

# ## Detecting Outliers

# ### Scatterplots

# In[42]:


import plotly.express as px

# Scatter plots for each column's values using plotly for interactivity
for column in columns:
    fig = px.scatter(df, x=column, y=column, title=f'Scatter Plot of {column}')
    fig.update_layout(
        width=900,  # You can adjust the width and height
        height=900,
        xaxis_title=column,
        yaxis_title=column
    )
    fig.show()


# ### Boxplots

# In[44]:


# Loop through each column and plot a boxplot
for col in columns:
    sns.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()


# ### IQR method for outliers

# In[46]:


import numpy as np

# Columns to analyze for outliers
columns = ['mpg', 'displayments', 'horsepower', 'weight', 'acceleration']

# Detect outliers using the IQR method
outliers = pd.DataFrame(index=df.index)  # Initialize outliers DataFrame
for col in columns:
    Q1 = df[col].quantile(0.25)  # 25th percentile
    Q3 = df[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile Range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)  # Mark outliers

# Combine outlier masks across all columns
outliers['is_outlier'] = outliers.any(axis=1)  # True if row is an outlier in any column

# Add an "outlier" column to the main DataFrame
df['outlier'] = np.where(outliers['is_outlier'], 'Outlier', 'Normal')

# Output a list of all outliers
outlier_list = df[df['outlier'] == 'Outlier']
outlier_list


# ## Categorical Values
# 
# Checking what car brands are seen most often in the dataset
# 
# 

# In[48]:


from collections import Counter
import re

if 'car name' in df.columns:
    car_names = df['car name']

    # tokenizing
    all_words = []
    for name in car_names:
        words = re.findall(r'\w+', name.lower())  # split and lowercase
        all_words.extend(words)

    # frequency of words
    word_counts = Counter(all_words)

    # most common
    most_common_words = word_counts.most_common(50)

most_common_words


# ### Brands

# In[50]:


# splitting last column, taking first token of 'car name' as a seperate column and the rest as the model of the car
df[['car brand', 'model']] = df['car name'].str.extract(r'^(\S+)\s*(.*)$')

df['car brand'].unique()


# In[51]:


brand_mapping = {
    'vw': 'volkswagen',
    'vokswagen': 'volkswagen',
    'chevy': 'chevrolet',
    'chevroelt': 'chevrolet',
    'mercedes-benz': 'mercedes',
    'mercedes': 'mercedes',
    'toyouta': 'toyota',
    'toyota': 'toyota',
     'maxda': 'mazda',
    'capri': 'mercury',

}

# standardize car brands
df['car brand'] = df['car brand'].replace(brand_mapping)

# recalculate the counts of car brands after mapping
car_brand_counts = df['car brand'].value_counts()

car_brand_counts


# In[52]:


# histogram of cars by brand
plt.figure(figsize=(12, 6))
car_brand_counts.plot(kind='bar', color='blue', edgecolor='black')
plt.title('Number of Cars by Brand', fontsize=16)
plt.xlabel('Car Brand', fontsize=12)
plt.ylabel('Number of Cars', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.tight_layout()
# plot
plt.show()


# In[53]:


# Filter the dataset to include only rows where 'mpg' is not missing
valid_data = df[df['mpg'].notna()]

# Set a threshold for the minimum number of instances per car brand (e.g., 10 instances)
min_instances = 10

# Get the count of instances per car brand
brand_counts = valid_data['car brand'].value_counts()

# Filter car brands that have at least 'min_instances' and valid 'mpg' values
valid_brands = brand_counts[brand_counts >= min_instances].index
filtered_data = valid_data[valid_data['car brand'].isin(valid_brands)]

# 1. Descriptive statistics: Mean MPG by Car Brand (excluding missing values and small brands)
mpg_by_brand = filtered_data.groupby('car brand')['mpg'].mean().sort_values(ascending=False)
print("Mean MPG by Car Brand (with enough instances and valid MPG):")
print(mpg_by_brand)

plt.figure(figsize=(12, 6))
sns.boxplot(data=filtered_data, x='car brand', y='mpg')
plt.title('Distribution of MPG by Car Brand (with enough instances and valid MPG)')
plt.xlabel('Car Brand')
plt.ylabel('MPG')
plt.xticks(rotation=90)
plt.show()



# We created boxplots to visualize the distribution of MPG across different car brands.However, to ensure meaningful analysis, we focused only on car brands with more than 10 instances. This was critical because analyzing brands with very few data points could lead to unreliable or misleading conclusions.

# Volkswagen, Mazda, Datsun and Honda showing the highest average MPG values, while brands like Ford, Mercury, Buick, and Amc tend to have lower MPG values.
# 

# In[56]:


# Descriptive statistics: Mean MPG by Origin
mpg_by_origin = valid_data.groupby('origin')['mpg'].mean().sort_values(ascending=False)
car_count_per_origin = valid_data['origin'].value_counts()
print("Mean MPG by Origin and Number of Cars per Origin:")
print(mpg_by_origin, car_count_per_origin)

#  Boxplot: Distribution of MPG by Origin
plt.figure(figsize=(12, 6))
sns.boxplot(data=valid_data, x='origin', y='mpg')
plt.title('Distribution of MPG by Car Origin')
plt.xlabel('Origin')
plt.ylabel('MPG')
plt.xticks(rotation=0)
plt.show()




# Origin 3 (Japan) likely corresponds to more fuel-efficient cars on average, while Origin 1 (USA) cars tend to have lower MPG. Origin 2 (Europe) lies in the between

# In[58]:


# saving to a new file 
df.to_csv('C:/Users/Katerina/Team2 Project/Team-2-Project/data/df_eda.csv', index=False)


# In[59]:


get_ipython().system('jupyter nbconvert --to script D1.ipynb')

