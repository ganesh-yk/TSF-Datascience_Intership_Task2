#!/usr/bin/env python
# coding: utf-8

# # Intern NAME: GANESH YK

# ## Task:To Explore Supervised Machine Learning
# 

# ## Task Objective :Student's Percentage Prediction Model

# ###### In this regression task we will predict the percentage of marks that a student is expected to score based upon the number of hours they studied. This is a simple linear regression task as it involves just two variables. 
# Data can be found at http://bit.ly/w-data
# 

# ###### Problem Statement: What will be predicted score if a student study for 9.25 hrs in a day? 

# # STEP:1 Importing the Required Libraries
# 

# In[19]:


# Importing all libraries required in this notebook
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


# In[3]:


#Import the data
url="http://bit.ly/w-data"
data=pd.read_csv(url)
data1=data
print("The data is imported successfully")
data


# Now we have to check whether the obtain data contains null value or not,if the data contains null value then data cleaning have to be done. 

# In[4]:


#checking for null values
data.isnull().sum()


# In[5]:


data.info()


# There are no null values and hence data cleaning is not required.

# In[36]:


data.describe()


# ## STEP:2 DATA VISUALIZATION
# Let's plot the obtained data for better understanding and Visualization.

# In[45]:


#Plotting the distribution of scores
data.plot(x='Hours', y='Scores', style='o',figsize=(16,8))  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[9]:


plt.style.use('seaborn-white')
plt.figure(figsize=(16,8))
sns.barplot(x='Hours',y='Scores',data=data)
plt.title('Study hours vs Scores obtained',size=18)
plt.xlabel('Study Hours',size=18)
plt.ylabel('Scores obtained',size=18)


# # STEP:3 Linear Regression Model
# 
# Now we prepare the data and split it in test data 

# In[12]:


#plotting regressor plot to determine the relationship between feature and target
plt.figure(figsize=(16,8))
sns.regplot(x=data['Hours'],y=data['Scores'],data=data,color='blue')
plt.title('Study Hours vs Percentage Scores')
plt.xlabel('Study Hours')
plt.ylabel('Percentage')
plt.show()


# ###### From the graph above, it can be clearly seen that there is a positive linear relation between the number of hours studied and percentage of score.

# ###### Preparing our data
# Next is to define our "attributes"(input) variable and "labels"(output)

# In[14]:


x=data.iloc[:,:-1].values  #Attributes
y=data.iloc[:,1].values   #Labels


# Now that we have the attributes and labels defined, the next step is to split this data into training and test sets.

# We will use sklearn's train_test_split to split the data into testing and training set

# In[15]:


# Using Scikit-Learn's built-in train_test_split() method:
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.80,test_size=0.20,random_state=0)


# We have the training and testing sets ready for training our model.
# 

# # STEP:4 Training the model
# 

# First I will be making our linear regression algorithm from scratch and then I will compare it with the built-in function sklearn.linear_model.LinearRegression()

# #### Making the linear regression from scratch

# In[20]:


linearRegressor= LinearRegression()
linearRegressor.fit(x_train, y_train)
y_predict= linearRegressor.predict(x_train)


# In[21]:


regressor = LinearRegression()  
regressor.fit(x_train, y_train) 
print("Training Process is completed.")


# ###### Now once the data are fitted to the model, we can try to plot the best fit line(regression line) which has less error.**

# Finding the coefficient  of the data

# In[22]:


print(regressor.coef_)


# Finding the intercept of the data

# In[23]:


print(regressor.intercept_)


# In[47]:


# Plotting the regression line
line = regressor.coef_*x+regressor.intercept_
# Plotting for the test data
plt.figure(figsize=(16,8))
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# # STEP:5 Checking the accuracy scores for training and test set

# In[30]:


print('Test Score:',regressor.score(x_test, y_test))
print('Training Score:',regressor.score(x_train, y_train))


# ## Now we make predictions

# In[33]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores


# In[34]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[35]:


#Let's predict the score for 9.25 hpurs
print('Score of student who studied for 9.25 hours a dat', regressor.predict([[9.25]]))


# # STEP:6 Model Evaluation Metrics

# In[46]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R-2:', metrics.r2_score(y_test, y_pred))

