#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# import plotly.express as px
import scipy.stats as stats
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Correct path example (update this path based on the above output)
df = pd.read_csv(r"C:\Users\akash\OneDrive\Documents\dataset\kaggle dataset\Imdb_indian_movies.zip")
df.head(10)


# In[3]:


df.info()


# In[4]:


df.columns


# In[5]:


df.describe()


# ##### Getting Information about Nan / Missing Values

# In[6]:


print("unique count")
print(df.nunique())
print("Null count")
df.isnull().sum()


# In[7]:


def missing_values_percent(dataframe):
    missing_values = dataframe.isna().sum()
    percentage_missing = (missing_values / len(dataframe) * 100).round(2)

    result_movie = pd.DataFrame({'Missing Values': missing_values, 'Percentage': percentage_missing})
    result_movie['Percentage'] = result_movie['Percentage'].astype(str) + '%'

    return result_movie


result = missing_values_percent(df)
result


# In[8]:


sns.heatmap(df.isnull())


# In[9]:


print(max(df['Duration']))
print(min(df['Duration']))
print(np.mean(df['Duration']))
print(max(df['Duration']))


# In[10]:


df['Duration']=df['Duration'].astype(int)
sns.boxplot(x=df['Duration'])


# In[11]:


#Duration mainly depends on Genre , Actors name and Director
# plt.figure(figsize=(106, 10))
sns.jointplot(data=df, y='Genre', x='Duration',height=69,dropna=True,kind='hist')
plt.xticks(rotation=45)
plt.show()


# In[12]:


median_duration_by_genre = df.groupby('Genre')['Duration'].median()

# Display median duration by genre
print("\nMedian duration by genre:")
print(median_duration_by_genre)

# Debug: Print rows with Duration == 0 before replacement
print("\nRows with Duration == 0 before replacement:")
print(df[df['Duration'] == 0])

# Replace Duration == 0 with median duration by genre inplace
for genre, median_duration in median_duration_by_genre.items():
    df.loc[(df['Duration'] == 0) & (df['Genre'] == genre), 'Duration'] = median_duration

# Debug: Print rows with Duration == 0 after replacement
print("\nRows with Duration == 0 after replacement:")
print(df['Duration'].value_counts())


# In[13]:


actors=['Actor 1','Actor 2',  'Actor 3']
for actor in actors:
    median_duration_by_actor = df.groupby(actor)['Duration'].median()

    for act, median_duration in median_duration_by_actor.items():
        df.loc[(df['Duration'] == 0) & (df[actor] == act), 'Duration'] = median_duration
    print(f"\nCount of each Duration value: when grouping by {actor}")
    print(df['Duration'].value_counts())



# In[14]:


df['Duration']=df['Duration'].astype(int)
sns.boxplot(x=df['Duration'])
print(max(df['Duration']))


# In[15]:


df=df.dropna(subset=['Duration'],axis=0)
df=df[df['Duration'] >= 60]
df.head()


# In[16]:


sns.displot(df['Duration'])


# In[17]:


df.info()


# In[18]:


sns.displot(df['Rating'])


# In[19]:


df=df.dropna(subset=['Rating'],axis=0)
result = missing_values_percent(df)
result


# In[20]:


df['Genre'] = df['Genre'].str.split(', ')
df = df.explode('Genre')
df['Genre'].fillna(df['Genre'].mode()[0], inplace=True)
df.head()


# In[21]:


df=df.dropna(subset=['Actor 1','Actor 2','Actor 3'],axis=0)
result = missing_values_percent(df)
result
df.info()


# In[22]:


# fig_year = px.histogram(df, x='Year', histnorm='probability density', nbins=30)
# fig_year.update_traces(selector=dict(type='histogram'))
# fig_year.update_layout(
#     title='Distribution of Year',
#     title_x=0.5,
#     title_pad=dict(t=20),
#     title_font=dict(size=20),
#     xaxis_title='Year',
#     yaxis_title='Probability Density',
#     xaxis=dict(showgrid=False),
#     yaxis=dict(showgrid=False),
#     bargap=0.02,
#     plot_bgcolor='white')


# In[23]:


# fig_dur = px.histogram(df, x = 'Duration', histnorm='probability density', nbins = 40)
# fig_dur.update_traces(selector=dict(type='histogram'))
# fig_dur.update_layout(
#     title='Distribution of Duration',
#     title_x=0.5, title_pad=dict(t=20),
#     title_font=dict(size=20), xaxis_title='Duration',
#     yaxis_title='Probability Density',
#     xaxis=dict(showgrid=False),
#     yaxis=dict(showgrid=False),
#     bargap=0.02,
#     plot_bgcolor = 'white')
# fig_dur.show()


# In[24]:


# fig_rat = px.histogram(df, x = 'Rating', histnorm='probability density', nbins = 40)
# fig_rat.update_traces(selector=dict(type='histogram'))
# fig_rat.update_layout(title='Distribution of Rating',
#                       title_x=0.5,
#                       title_pad=dict(t=20),
#                       title_font=dict(size=20),
#                       xaxis_title='Rating',
#                       yaxis_title='Probability Density',
#                       xaxis=dict(showgrid=False),
#                       yaxis=dict(showgrid=False),
#                       bargap=0.02,
#                       plot_bgcolor = 'white')
# fig_rat.show()


# In[24]:


# fig_vot = px.box(df, x = 'Votes')
# fig_vot.update_layout(title='Distribution of Votes',
#                         title_x=0.5,
#                         title_pad=dict(t=20),
#                         title_font=dict(size=20),
#                         xaxis_title='Votes',
#                         yaxis_title='Probability Density',
#                         xaxis=dict(showgrid=False),
#                         yaxis=dict(showgrid=False),
#                         plot_bgcolor = 'white')
# fig_vot.show()


# In[25]:


# rel_dur_rat = px.scatter(df, x = 'Duration', y = 'Rating', color = "Rating")
# rel_dur_rat.update_layout(title='Rating v/s Duration of Movie',
#                           title_x=0.5,
#                           title_pad=dict(t=20),
#                           title_font=dict(size=20),
#                           xaxis_title='Duration of Movie in Minutes',
#                           yaxis_title='Rating of a movie',
#                           xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=False),
#                           plot_bgcolor = 'white')
# rel_dur_rat.show()


# In[26]:


# rel_dur_rat = px.scatter(df, x = 'Actor 1', y = 'Rating', color = "Rating")
# rel_dur_rat.update_layout(title='Rating v/s Actor 1',
#                           title_x=0.5,
#                           title_pad=dict(t=20),
#                           title_font=dict(size=20),
#                           xaxis_title='Actor 1',
#                           yaxis_title='Rating of a movie',
#                           xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=False),
#                           plot_bgcolor = 'white')
# rel_dur_rat.show()


# In[27]:


# rel_dur_rat = px.scatter(df, x = 'Actor 2', y = 'Rating', color = "Rating")
# rel_dur_rat.update_layout(title='Rating v/s Actor 2',
#                           title_x=0.5,
#                           title_pad=dict(t=20),
#                           title_font=dict(size=20),
#                           xaxis_title='Actor 2',
#                           yaxis_title='Rating of a movie',
#                           xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=False),
#                           plot_bgcolor = 'white')
# rel_dur_rat.show()


# In[28]:


# rel_dur_rat = px.scatter(df, x = 'Actor 3', y = 'Rating', color = "Rating")
# rel_dur_rat.update_layout(title='Rating v/s Actor 3',
#                           title_x=0.5,
#                           title_pad=dict(t=20),
#                           title_font=dict(size=20),
#                           xaxis_title='Actor 3',
#                           yaxis_title='Rating of a movie',
#                           xaxis=dict(showgrid=False),
#                           yaxis=dict(showgrid=False),
#                           plot_bgcolor = 'white')
# rel_dur_rat.show()


# In[30]:

#
# fig_rat_votes = px.scatter(df, x = 'Rating', y = 'Votes', color = "Votes")
# fig_rat_votes.update_layout(title='Getting Look at  Ratings impact on Votes ',
#                             title_x=0.5,
#                             title_pad=dict(t=20),
#                             title_font=dict(size=20),
#                             xaxis_title='Ratings of Movies',
#                             yaxis_title='Votes of movies',
#                             xaxis=dict(showgrid=False),
#                             yaxis=dict(showgrid=False),
#                             plot_bgcolor = 'white')
# fig_rat_votes.show()


# In[31]:


df.drop('Name', axis = 1, inplace = True)


# ### Feature Engineering

# In[32]:


g_mean_rat = df.groupby('Genre')['Rating'].transform('mean')
df['G_mean_rat'] = g_mean_rat

dir_mean_rat = df.groupby('Director')['Rating'].transform('mean')
df['Dir_enc'] = dir_mean_rat

a1_mean_rat = df.groupby('Actor 1')['Rating'].transform('mean')
df['A1_enc'] = a1_mean_rat

a2_mean_rat = df.groupby('Actor 2')['Rating'].transform('mean')
df['A2_enc'] = a2_mean_rat

a3_mean_rat = df.groupby('Actor 3')['Rating'].transform('mean')
df['A3_enc'] = a3_mean_rat


# In[33]:


df.head(10)


# ### Splitng Data

# In[37]:


X = df[['Year', 'Votes', 'Duration', 'G_mean_rat', 'Dir_enc', 'A1_enc', 'A2_enc', 'A3_enc']]
y = df['Rating']
print(X.shape)
print(y.shape)


# In[38]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
X_train.info()


# ### Models Used :
# - Linear Regression
# - decision Tree Regressor
# - Random Forest Regressor ## Techniques used for Varification Of data
# - R2 Score
# - K-Fold cross-
# - Mean squared error
# - Mean absolute error:

# ### Applying Linear Regression and Predicting

# In[39]:


lr = LinearRegression()
lr.fit(X_train,y_train)
lr_pred = lr.predict(X_test)


# In[40]:


print('The performance evaluation of Linear Regression is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, lr_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, lr_pred))
print('R2 score:', metrics.r2_score(y_test, lr_pred))
print('\n', '='*100, '\n')

# Perform 5-fold cross-validation for Linear Regression
cv_scores_lr = cross_val_score(lr, X, y, cv=8, scoring='r2')
print('Linear Regression 5-fold cross-validation R2 scores:', cv_scores_lr)
print('Mean R2 score:', cv_scores_lr.mean())


# ### Applying Decision Tree Regressor and Predicting

# In[41]:


dt_regressor = DecisionTreeRegressor(random_state=2)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)


# In[42]:


print('The performance evaluation of Decision Tree Regressor is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))

# Perform 5-fold cross-validation for Decision Tree Regressor
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=8, scoring='r2')
print('\nDecision Tree Regressor 5-fold cross-validation R2 scores:', cv_scores_dt)
print('Mean R2 score:', cv_scores_dt.mean())


# In[ ]:





# In[33]:


dt_regressor = DecisionTreeRegressor(random_state=2,max_depth=14)
dt_regressor.fit(X_train, y_train)
y_pred = dt_regressor.predict(X_test)


# In[43]:


print('The performance evaluation of Decision Tree Regressor is below:')
print('Mean squared error:', metrics.mean_squared_error(y_test, y_pred))
print('Mean absolute error:', metrics.mean_absolute_error(y_test, y_pred))
print('R2 score:', metrics.r2_score(y_test, y_pred))

# Perform 5-fold cross-validation for Decision Tree Regressor
cv_scores_dt = cross_val_score(dt_regressor, X, y, cv=8, scoring='r2')
print('\nDecision Tree Regressor 5-fold cross-validation R2 scores:', cv_scores_dt)
print('Mean R2 score:', cv_scores_dt.mean())


# In[44]:


from sklearn.tree import plot_tree


# In[45]:


plot_tree(dt_regressor)


# In[ ]:





# In[ ]:





# ## Conclusion
# - Decision Tree Regressor: Good performance with MSE of 0.088 and R² of 0.952, but cross-validation shows variability.
# - Linear Regression: Lowest performance with MSE of 0.481 and R² of 0.780, though consistent cross-validation scores.

# In[ ]:

import pickle

# Assume dt_regressor is your trained model
with open("decision_tree_model.pkl", "wb") as file:
    pickle.dump(dt_regressor, file)



