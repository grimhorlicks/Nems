#!/usr/bin/env python
# coding: utf-8

# # Customer Churn Rate Prediction
# ### Nemisha

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_excel("/Users/nemisha/Desktop/new/customer_churn_large_dataset.xlsx")


# In[3]:


df.head()


# ## Data Preprocessing

# In[4]:


df.info()


# In[5]:


df.isnull().sum() 


# The data has no missing value and is ready for exploratorty data analysis
# 

# In[6]:


df.shape


# In[7]:


df.columns.values


# In[8]:


df.describe()


# ## EDA

# In[9]:


# for plotting 
import plotly.express as px
import pandas as pd
import plotly
import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


df['Gender'].value_counts()


# In[11]:


list_count_by_Gender=df['Gender'].value_counts()
list_count_by_Gender_list=list(list_count_by_Gender.index)


# In[12]:


fig = px.pie(values=list_count_by_Gender, names=list_count_by_Gender.index)
fig.show()


# In[13]:


df['Location'].value_counts()


# In[14]:


list_count_by_Location=df['Location'].value_counts()
list_count_by_Location_list=list(list_count_by_Location.index)
fig = px.pie(values=list_count_by_Location, names=list_count_by_Location.index)
fig.show()


# In[15]:


sns.displot(data=df,x='Age',kde=True,discrete=True)


# In[16]:


ages = pd.DataFrame(df, columns=['Age'])

bins = [18, 30, 40, 50, 60, 70, 120]
labels = ['18-29', '30-39', '40-49', '50-59', '60-69', '70+']
ages['agerange'] = pd.cut(ages.Age, bins, labels = labels,include_lowest = True)

ages


# In[17]:


ages['agerange'].value_counts()


# In[18]:


list_count_by_age=ages['agerange'].value_counts()
list_count_by_age_list=list(list_count_by_age.index)


# In[19]:


list_count_by_age_list


# In[20]:


fig= px.bar(ages, x=list_count_by_age.index ,y=list_count_by_age) 
fig.show()


# In[21]:


df['Churn'].value_counts()


# In[22]:


list_count_by_Churn=df['Churn'].value_counts()
list_count_by_Churn_list=list(list_count_by_Churn.index)
fig = px.pie(values=list_count_by_Churn, names=list_count_by_Churn.index)
fig.show()


# In[23]:


df["Churn"][df["Churn"]==1].groupby(by=df["Gender"]).count() #1=yes churn


# In[24]:


df["Churn"][df["Churn"]==0].groupby(by=df["Gender"]).count() #0=no churn


# In[25]:


plt.figure(figsize=(6, 6))
labels =["Churn:1","Churn:0"]
values = [49779,50221]
labels_gender = ["F","M","F","M"]
sizes_gender = [24944,24835 , 25272,24949]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0','#ffb3e6', '#c2c2f0','#ffb3e6']
explode = (0.3,0.3) 
explode_gender = (0.1,0.1,0.1,0.1)
textprops = {"fontsize":15}
#Plot
plt.pie(values, labels=labels,autopct='%1.1f%%',pctdistance=1.08, labeldistance=2.5,colors=colors, startangle=90,frame=True, explode=explode,radius=10, textprops =textprops, counterclock = True, )
plt.pie(sizes_gender,labels=labels_gender,colors=colors_gender,startangle=90, explode=explode_gender,radius=7, textprops =textprops, counterclock = True, )
#Draw circle
centre_circle = plt.Circle((0,0),5,color='black', fc='white',linewidth=0)
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# show plot 
 
plt.axis('equal')
plt.tight_layout()
plt.show()


# In[26]:


sub_length = pd.DataFrame(df, columns=['Subscription_Length_Months','Churn'])

bins = [1, 4, 8, 12, 16, 20 , 24]
labels = ['1-4', '5-8', '9-12', '13-16', '17-20', '21-24']
sub_length['Month_interval'] = pd.cut(sub_length.Subscription_Length_Months, bins, labels = labels,include_lowest = True)

sub_length


# In[27]:


fig = px.histogram(sub_length, x="Churn", color="Month_interval", barmode="group", title="<b>Customer Subscription Length<b>")
fig.update_layout(width=700, height=500, bargap=0.1)
fig.show()


# In[28]:


df


# In[29]:


sns.displot(data=df,x='Monthly_Bill',kde=True,discrete=True)


# In[30]:


ax = sns.kdeplot(df.Monthly_Bill[(df["Churn"] == 0) ],
                color="Gold", shade = True);
ax = sns.kdeplot(df.Monthly_Bill[(df["Churn"] == 1) ],
                ax =ax, color="Green", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Monthly_Bill');
ax.set_title('Distribution of Monthly_Bill by churn');


# In[31]:


sns.displot(data=df,x='Total_Usage_GB',kde=True,discrete=True)


# In[32]:


ax = sns.kdeplot(df.Total_Usage_GB[(df["Churn"] == 0) ],
                color="Gold", shade = True);
ax = sns.kdeplot(df.Total_Usage_GB[(df["Churn"] == 1) ],
                ax =ax, color="Green", shade= True);
ax.legend(["Not Churn","Churn"],loc='upper right');
ax.set_ylabel('Density');
ax.set_xlabel('Total_Usage_GB');
ax.set_title('Distribution of Total_Usage_GB by churn');


# In[33]:


fig = px.box(df, x='Churn', y = 'Subscription_Length_Months')

# Update yaxis properties
fig.update_yaxes(title_text='Subscription_Length_Months (Months)', row=1, col=1)
# Update xaxis properties
fig.update_xaxes(title_text='Churn', row=1, col=1)

# Update size and title
fig.update_layout(autosize=True, width=750, height=600,
    title_font=dict(size=25, family='Courier'),
    title='<b>Subscription_Length_Months vs Churn</b>',
)

fig.show()


# In[34]:


plt.figure(figsize=(25, 10))

corr = df.apply(lambda x: pd.factorize(x)[0]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)


# In[35]:


plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)


# In[36]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[37]:


data = pd.get_dummies(df, columns = ['Gender'])
print(data)


# In[38]:


data


# In[39]:


data.drop(['Name'], axis=1)


# ### Feature scalling
# 

# In[40]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature], color= color)


# In[41]:


num_cols = ["Subscription_Length_Months", 'Monthly_Bill', 'Total_Usage_GB']
for feat in num_cols: distplot(feat, data)


# In[42]:


df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in num_cols: distplot(feat, df_std, color='c')


# ## Machine Learning Model Evaluations and Predictions

# In[43]:


import gc
import warnings
# Sklearn imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
warnings.simplefilter(action='ignore', category=FutureWarning)


# In[44]:


from sklearn.ensemble import RandomForestClassifier


# ## Logistics regression

# In[45]:


data.columns.values


# In[46]:


features = data[['CustomerID', 'Age',
       'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB',
          'Gender_Female', 'Gender_Male']]

labels = data['Churn']


# In[47]:


train_df = data[:int(len(df)*0.8)]
val_df = data[int(len(df)*0.8):]


# In[48]:


print('\nData in Train:')
print(train_df['Churn'].value_counts())

print('\nData in Val:')
print(val_df['Churn'].value_counts())


# #### Balancing the data set

# In[49]:


class_0 = train_df[train_df['Churn'] == 0]
class_1 = train_df[train_df['Churn'] == 1]

class_1 = class_1.sample(len(class_0),replace=True)
train_df = pd.concat([class_0, class_1], axis=0)
print('Data in Train:')
print(train_df['Churn'].value_counts())


# In[50]:


class_0 = val_df[val_df['Churn'] == 0]
class_1 = val_df[val_df['Churn'] == 1]

class_1 = class_1.sample(len(class_0),replace=True)
val_df = pd.concat([class_0, class_1], axis=0)
print('Data in Test:')
print(val_df['Churn'].value_counts())


# In[51]:


#Model
x_train = np.array(train_df[['CustomerID', 'Age',
       'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB',
          'Gender_Female', 'Gender_Male']])
y_train = np.array(train_df['Churn'])

x_val = np.array(val_df[['CustomerID', 'Age',
       'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB',
          'Gender_Female', 'Gender_Male']])
y_val = np.array(val_df['Churn'])


# In[52]:


model = LogisticRegression().fit(x_train, y_train)


# In[53]:


y_pred = model.predict(x_val)


# In[54]:


print(classification_report(y_val,y_pred))


# In[55]:


cm = confusion_matrix(y_val, y_pred)

plt.figure(figsize=(7,5))

ax = sns.heatmap(cm/np.sum(cm),fmt='.2%', annot=True, cmap='Blues')

ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['No Churn','Churn'])
ax.yaxis.set_ticklabels(['No Churn','Churn'])

plt.show()


# In[56]:


y_pred = model.predict_proba(x_val)[:,1]
fpr, tpr, thresholds = roc_curve(y_val, y_pred)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show();


# ## KNN Model

# In[57]:


knn_model = KNeighborsClassifier(n_neighbors = 11) 


# In[58]:


knn_model = knn_model.fit(x_train, y_train)


# In[59]:


predicted_y = knn_model.predict(x_val)


# In[60]:


accuracy_knn = knn_model.score(x_val,y_val)


# In[61]:


print("KNN accuracy:",accuracy_knn)


# In[62]:


print(classification_report(y_val, predicted_y))


# ### Saving the model

# In[63]:


pip install joblib 


# In[70]:


import sys


# In[71]:


import joblib

sys.modules['sklearn.externals.joblib'] = joblib


# In[73]:


joblib_file = "joblib_knn_model.pkl"  
joblib.dump(knn_model, joblib_file)


# In[74]:


# Load from file

joblib_knn_model = joblib.load(joblib_file)


joblib_knn_model


# In[75]:


# Use the Reloaded Joblib Model to 
# Calculate the accuracy score and predict target values

# Calculate the Score 
score = joblib_knn_model.score(x_val,y_val)  
# Print the Score
print("Test score: {0:.2f} %".format(100 * score))  

# Predict the Labels using the reloaded Model
predicted_y = joblib_knn_model.predict(x_val)  

predicted_y


# ## SVC

# In[ ]:


svc_model = SVC(random_state = 1)
svc_model.fit(x_train, y_train)
predict_y = svc_model.predict(x_val)
accuracy_svc = svc_model.score(x_val,y_val)
print("SVM accuracy is :",accuracy_svc)


# In[ ]:


print(classification_report(y_val, predict_y))


# ## Random Forest

# In[ ]:


model_rf = RandomForestClassifier(n_estimators=500 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(x_train, y_train)

# Make predictions
prediction_test = model_rf.predict(x_val)
print (metrics.accuracy_score(y_val, prediction_test))


# In[ ]:


print(classification_report(y_val, prediction_test))


# In[ ]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_val, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[ ]:


y_rfpred_prob = model_rf.predict_proba(x_val)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_val, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='Random Forest',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();


# In[ ]:




