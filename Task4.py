#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Load the dataset
data = pd.read_csv("sample_traffic_accident_data.csv")


# In[3]:


# Display the first few rows of the dataset
print("Data Preview:\n", data.head())


# In[4]:


# Data Cleaning (Removing null values, duplicates, etc.)
data = data.dropna()  # Drop rows with missing values
data = data.drop_duplicates()  # Remove duplicate entries


# In[5]:


# Data Summary
print("\nSummary Statistics:\n", data.describe())


# In[6]:


# Exploratory Data Analysis (EDA)
# 1. Accident Severity Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Accident_Severity', data=data, palette="coolwarm")
plt.title("Distribution of Accident Severity")
plt.xlabel("Severity")
plt.ylabel("Count")
plt.show()


# In[7]:


# 2. Accidents by Weather Conditions
plt.figure(figsize=(8, 5))
sns.countplot(x='Weather_Conditions', data=data, palette="Blues_d")
plt.title("Accidents by Weather Conditions")
plt.xlabel("Weather")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[8]:


# 3. Accidents by Time of Day
data['Time'] = pd.to_datetime(data['Time'], format='%H:%M:%S')
data['Hour'] = data['Time'].dt.hour
plt.figure(figsize=(10, 6))
sns.histplot(data['Hour'], bins=24, kde=False, color='green')
plt.title("Accidents by Time of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Accident Count")
plt.show()


# In[9]:


# 4. Impact of Road Surface Condition on Accident Severity
plt.figure(figsize=(10, 6))
sns.boxplot(x='Road_Surface_Condition', y='Vehicle_Speed', hue='Accident_Severity', data=data, palette="Set2")
plt.title("Impact of Road Surface Condition on Vehicle Speed and Severity")
plt.xlabel("Road Surface Condition")
plt.ylabel("Vehicle Speed")
plt.show()


# In[10]:


# 5. Correlation Heatmap
corr_matrix = data[['Vehicle_Speed', 'Driver_Age', 'Injury_Count', 'Fatality_Count']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Between Numerical Features")
plt.show()


# In[12]:


# Generate a Summary Report
summary = data.groupby('Accident_Severity').agg({
    'Accident_ID': 'count',
    'Vehicle_Count': 'mean',
    'Vehicle_Speed': 'mean',
    'Injury_Count': 'sum',
    'Fatality_Count': 'sum'
}).reset_index()
print("\nSummary Report:\n", summary)


# In[ ]:




