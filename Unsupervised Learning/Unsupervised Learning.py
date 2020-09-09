#!/usr/bin/env python
# coding: utf-8

# # Intern NAME:GANESH YK

# ## TASK3: To Explore Unsupervised Machine Learning

# ###### From the given ‘Iris’ dataset, predict the optimum number of clusters and represent it visually.
# 

# DataSet: https://drive.google.com/file/d/11Iq7YvbWZbt8VXjfm06brx66b10YiwK-/view?usp=sharing

# In[1]:


# Importing the Required libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets


# In[3]:


# Reading the data 
iris_df = pd.read_csv("Iris.csv", index_col = 0)
print("Let's see a part of the whole dataset - \n")
iris_df.head()  # See the first 5 rows


# In[4]:


print ("The info about the datset is as follows - \n")
iris_df.info()


# As we can see there are no null values present. We can use the dataset as it is.
# Let's plot a pair plot to visualise all the attributes's dependency on each other in one go.

# In[5]:


# Plotting the pair plot
sns.pairplot(iris_df, hue = 'Species')


# In[6]:


# Defining 'X'
X = iris_df.iloc[:, [0, 1, 2, 3]].values


# In[16]:


# Finding the optimum number of clusters for k-means classification
from sklearn.cluster import KMeans
wcss = [] 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
# Plotting the results onto a line graph, 
# Allowing us to observe 'The elbow'
plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') # Within cluster sum of squares
plt.show()
sns.set(rc={'figure.figsize':(7,5)})


# It can be clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as **3**.

# In[8]:


# Applying kmeans to the dataset / Creating the kmeans classifier
kmeans = KMeans(n_clusters = 3, init = 'k-means++',
                max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)
y_kmeans


# In[19]:


# Visualising the clusters - On the first two columns
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], 
            s = 100, c = 'blue', label = 'Iris-setosa')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], 
            s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')
# Plotting the centroids of the clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], 
            s = 100, c = 'red', label = 'Centroids')
plt.legend()

sns.set(rc={'figure.figsize':(16,8)})


# In[ ]:




