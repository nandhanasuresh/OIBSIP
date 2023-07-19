#!/usr/bin/env python
# coding: utf-8

# # Iris Flower Classification Dataset
# 
# 

# ### The Iris flower classification dataset comprises 150 samples of Iris flowers, categorized into three species:
# 
# 1. Iris setosa
# 2. Iris versicolor
# 3. Iris virginica
# 
# 
# ### The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.One class is linearly separable from the other 2; the latter are NOT linearly separable from each other.
# 
# Attribute Information:
# 1. Sepal length in cm
# 2. sepal width in cm
# 3. petal length in cm
# 4. petal width in cm
# 5. class
# 
# ### The dataset is widely used as a benchmark in machine learning for supervised classification tasks aiming to accurately classify Iris flowers based on their measurements.

# ## Import Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# 
# ## Load the dataset

# In[2]:


df = pd.read_csv('Iris.csv')
df.head()


# In[3]:


# Drop the 'Id' column as it is not required for analysis
df = df.drop(columns=["Id"])
df.head()


# In[4]:


#Display the first 10 rows of the dataframe
df.head(10)


# In[5]:


# Display basic statistics about the data
df.describe()


# In[6]:


# Display information about the datatype of each column and null values
df.info()


# In[7]:


# Display the number of samples for each class
df['Species'].value_counts()


# ## Preprocessing the Dataset

# In[8]:


#Label encoding to convert class labels into numeric form
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])
df['Species']


# In[9]:


df


# In[10]:


# check for null values
df.isnull().sum()


# ### Exploratory Data Analysis (EDA)
# 

# In[11]:


# Plot histograms of each feature
df['SepalLengthCm'].hist()


# In[12]:


df['SepalWidthCm'].hist()


# In[13]:


df['PetalLengthCm'].hist()


# In[14]:


df['PetalWidthCm'].hist()


# In[15]:


#Plotting the histogram of all features toghether
df['SepalLengthCm'].hist()
df['SepalWidthCm'].hist()
df['PetalLengthCm'].hist()
df['PetalWidthCm'].hist()


# In[16]:


# Plot scatterplots to visualize relationships between features
colors = ['red', 'orange', 'blue']
species = [0, 1, 2]


# In[17]:


# Scatter plot for Sepal Length vs Sepal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# In[18]:


# Scatter plot for Petal Length vs Petal Width 
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.legend()


# In[19]:


# Scatter plot for Petal Length vs Sepal Length
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.legend()


# In[20]:


# Scatter plot for Sepal Width vs Petal Width
for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c = colors[i], label=species[i])
plt.xlabel("Sepal Width")
plt.ylabel("Petal Width")
plt.legend()


# ## Correlation Matrix

# A correlation matrix is a table showing correlation coefficients between variables. Each cell in the table shows the correlation between two variables. The value is in the range of -1 to 1. If two varibles have high correlation, we can neglect one variable from those two.

# In[21]:


# Compute the correlation matrix 
df.corr()


# In[22]:


# display the correlation matrix using a heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')


# ## Model Training

# In[23]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.40)


# In[24]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
print("Accuracy (Logistic Regression): ", model1.score(x_test, y_test) * 100)


# In[25]:


# K-nearest Neighbours Model (KNN)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train)
print("Accuracy (KNN): ", model2.score(x_test, y_test) * 100)


# In[26]:


# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
print("Accuracy (Decision Tree): ", model3.score(x_test, y_test) * 100)


# ## Confusion Matrix

# In[27]:


from sklearn.metrics import confusion_matrix


# In[28]:


y_pred1 = model1.predict(x_test)
y_pred2 = model2.predict(x_test)
y_pred3 = model3.predict(x_test)


# In[29]:


conf_matrix1 = confusion_matrix(y_test, y_pred1)
conf_matrix2 = confusion_matrix(y_test, y_pred2)
conf_matrix3 = confusion_matrix(y_test, y_pred3)


# In[30]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix1, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Logistic Regression')
plt.show()


# In[31]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix2, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of KNN')
plt.show()


# In[32]:


plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix3, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(Y), yticklabels=np.unique(Y))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix of Decision Tree')
plt.show()


# In[ ]:





# In[ ]:




