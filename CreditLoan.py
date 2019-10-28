#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install pandas
import pandas as pd

#!pip install warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("LoanStats3c.csv")
df1 = pd.read_csv("LoanStats3c.csv")
pd.set_option('display.max_columns', len(df.columns))
df.head()


# In[2]:


df.info()


# In[3]:


df.term.value_counts()


# In[4]:


df['term'] = df['term'].apply(lambda x:x.replace("months" , ''))
df['term'] = pd.Categorical(df['term'])
# term is in the rate of months , since it is either 36 or 60 months
# we can make it into a categorical value
df['int_rate'] = df["int_rate"].apply(lambda x:x.replace('%',''))
df['int_rate'] = pd.to_numeric(df['int_rate'])
df['int_rate'] = (df['int_rate'])/100
# int_rate is in the rate of percentage
df.head(2)


# In[5]:


df.grade.value_counts()


# In[6]:


#!pip install sklearn

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df.grade = le.fit_transform(df.grade)
df.grade = pd.Categorical(df.grade)

# A = 0
# B = 1
# C = 2
# D = 3
# E = 4
# F = 5
# G = 6


# In[7]:


le = preprocessing.LabelEncoder()
df.sub_grade = le.fit_transform(df.sub_grade)
df.sub_grade = pd.to_numeric(df.sub_grade)


# In[8]:


# !pip install numpy

import numpy as np

print(len(np.unique(df.member_id)) == len(df))

# Every member ID is unique so we can use it as our index

df.index = df['member_id']


# In[9]:


(df.loan_amnt.corr(df.funded_amnt) , df.loan_amnt.corr(df.funded_amnt_inv) , df.out_prncp.corr(df.out_prncp_inv))

# Loan amount and funded amount is the same column so we can drop one of them
# Loan amount and funded_amnt_inv is extremely close to one as well so we can drop it too
# Same for out_prncp and out_prncp_inv


# In[10]:


df.drop(["funded_amnt","funded_amnt_inv" , "mths_since_last_major_derog" , 
         "mths_since_last_record" , 'desc' , "emp_title", "title" ,
        "url" , "zip_code" , "addr_state", "id" , "member_id" , "last_pymnt_d"
        , "next_pymnt_d" , "last_credit_pull_d" , "issue_d" , 
        "earliest_cr_line" , 'out_prncp_inv'] , axis = 1 , inplace = True)


# In[11]:


df.home_ownership.value_counts()


# In[12]:


df.is_inc_v.value_counts()


# In[13]:


df.pymnt_plan.value_counts()


# In[14]:


df.initial_list_status.value_counts()


# In[15]:


# Cleaning the data, replacing the categorical strings to numerical values
# And changing it to a categorical variable in the dataframe

df.replace(["MORTGAGE" , 'RENT' , 'OWN' , 'ANY'] , [0,1,2,3] , inplace = True)
df.home_ownership = pd.Categorical(df.home_ownership)
df.replace(["Source Verified" , "Not Verified" , "Verified"] , [0,1,2] , inplace = True)
df.is_inc_v = pd.Categorical(df.is_inc_v)
df.pymnt_plan.replace(['n' , 'y'] , (0,1), inplace = True)
df.pymnt_plan = pd.Categorical(df.pymnt_plan)
df.initial_list_status.replace(['w' , 'f'] , (0,1) , inplace = True)
df.initial_list_status = pd.Categorical(df.initial_list_status)
df.policy_code = pd.Categorical(df.policy_code)
df.inq_last_6mths = pd.Categorical(df.inq_last_6mths)


# In[16]:


df.purpose.value_counts()


# In[17]:


# Best way to categoricalize these values is to use the get dummies function
# of pandas then concat it with the original dataset

df2 = (pd.get_dummies(df.purpose))
df2 = pd.DataFrame(df2)
df2 = df2.astype("category")
df3 = pd.concat([df,df2] , axis = 1)
df = df3
df.drop("purpose" , axis = 1 , inplace = True)


# In[18]:


df.head()


# In[19]:


df.loan_status.value_counts()


# In[20]:


df.loan_status.replace(['Fully Paid' , 'Current' , 'Issued' , 'In Grace Period'
                       , 'Late (16-30 days)' , 'Late (31-120 days', 'Charged Off'
                       , 'Default'] , ("A" , "B" , "C" , "D" , "E" , "F" , "G" , "H")
                      , inplace = True)
df.loan_status = le.fit_transform(df.loan_status)
df.loan_status = pd.Categorical(df.loan_status)


# In[21]:


df.isnull().sum()


# In[22]:


df.emp_length.value_counts()


# In[23]:


# I need a placeholder for the N/A values otherwise
# I cannot use the lambda function, so I will use "300 years" and "100000%" to fill N/A values

df.emp_length = df.emp_length.fillna("300 years")
df['emp_length'] = df["emp_length"].apply(lambda x:x.replace('years',''))
df['emp_length'] = df["emp_length"].apply(lambda x:x.replace('year',''))
df['emp_length'] = df["emp_length"].apply(lambda x:x.replace('10+','11'))
df['emp_length'] = df["emp_length"].apply(lambda x:x.replace('< 1','0.5'))
df['emp_length'] = pd.to_numeric(df['emp_length'])


# In[24]:


# I will now use linear regression to replace the placeholders
# Instead of dropping the missing values or replacing it with a mean or median
# The best way I believe is using linear regression to find the best educated
# guess for those missing values


# In[25]:


X_test = df[df.emp_length == 300].drop(["emp_length" , "mths_since_last_delinq"
                                       ,"revol_util"  ] , axis = 1)
X_train = df[df.emp_length != 300].drop(["emp_length" , "mths_since_last_delinq"
                                       ,"revol_util"  ] , axis = 1)
y_test = df.emp_length[df.emp_length == 300]
y_train = df.emp_length[df.emp_length != 300]


# In[26]:


# !pip install sklearn

from sklearn.linear_model import LinearRegression

linreg = LinearRegression()
linreg.fit(X_train , y_train)
y_pred = np.rint(linreg.predict(X_test))
abc = (y_pred)
df.emp_length[df.emp_length == 300] = abc


# In[27]:


df.revol_util = df.revol_util.fillna("100000%")
df['revol_util'] = df["revol_util"].apply(lambda x:x.replace('%',''))
df['revol_util'] = pd.to_numeric(df['revol_util'])


# In[28]:


X_test = df[df.revol_util == 100000].drop(["mths_since_last_delinq"
                                       ,"revol_util"  ] , axis = 1)
X_train = df[df.revol_util != 100000].drop(["mths_since_last_delinq"
                                       ,"revol_util"  ] , axis = 1)
y_test = df.revol_util[df.revol_util == 100000]
y_train = df.revol_util[df.revol_util != 100000]


# In[29]:


linreg = LinearRegression()
linreg.fit(X_train , y_train)
y_pred = np.rint(linreg.predict(X_test))
abc = (y_pred)
df.revol_util[df.revol_util == 100000] = abc
df.revol_util =( df.revol_util)/(100)


# In[30]:


df.mths_since_last_delinq = df.mths_since_last_delinq.fillna(10000000)

X_test = df[df.mths_since_last_delinq == 10000000].drop(["mths_since_last_delinq"] , axis = 1)
X_train = df[df.mths_since_last_delinq != 10000000].drop(["mths_since_last_delinq"] , axis = 1)
y_test = df.mths_since_last_delinq[df.mths_since_last_delinq == 10000000]
y_train = df.mths_since_last_delinq[df.mths_since_last_delinq != 10000000]


# In[31]:


linreg = LinearRegression()
linreg.fit(X_train , y_train)
y_pred = np.rint(linreg.predict(X_test))
abc = (y_pred)
df.mths_since_last_delinq[df.mths_since_last_delinq == 10000000] = abc


# In[32]:


df.head()


# In[33]:


df["monthly_loan_income_%"] = df.installment/(( df.annual_inc/12) 
                                              - (df.annual_inc/12)*(df.dti/100))
df["monthly_loan_income_%"] = np.round(df["monthly_loan_income_%"],4)
# Let's figure out the percentage of the monthly payment from their
# monthly income - dti
# this column is in DECIMAL FORM NOT PERCENTAGE FORM


# In[34]:


#Rearranging the columns so we can see them side by side

cols = list(df)
cols.insert(3, cols.pop(cols.index("annual_inc")))
cols.insert(5, cols.pop(cols.index("monthly_loan_income_%")))
df = df.ix[:, cols]


# In[35]:


df["revol_avail"] = np.rint(df.revol_bal - df.revol_bal*df.revol_util)

# Creating a new variable to see how much available revolving credit for the
# client to further examine the financial status of the client(s)


# In[36]:


cols = list(df)
cols.insert(6, cols.pop(cols.index("revol_bal")))
cols.insert(7, cols.pop(cols.index("revol_util")))
cols.insert(8, cols.pop(cols.index("revol_avail")))
df = df.ix[:, cols]
df.head()


# In[37]:


df["potential_risk"] = pd.Categorical(np.logical_or(np.logical_or(np.logical_or
                                     (np.logical_and(df.is_inc_v == 1  , 
                                                     df.delinq_2yrs >= 1),
                                      pd.to_numeric(df.loan_status) >= 4) , 
                                     df.recoveries > 0 ),pd.to_numeric(df.grade)>2).astype('uint8'))

np.logical_or(np.logical_or(np.logical_or
                                     (np.logical_and(df.is_inc_v == 1  , 
                                                     df.delinq_2yrs >= 1),
                                      pd.to_numeric(df.loan_status) >= 4), 
                                     df.recoveries > 0 ),pd.to_numeric(df.grade)>2).astype('uint8').value_counts()

# Creating a new variable "potential_risk":
# is the income not verified and did they have a delinquency in the past two yrs
# OR is their loan status a 4 or higher OR was their recoveries greater than 0
# OR is their loan grade higher than a C?

# Banks and other financial institutes should be aware of those that can be a 
# potential risk which I think is a imperative variable for this dataset.

# 59203 people are potential risks based on my criteria


# In[38]:


df["red_flag"] = pd.Categorical(np.logical_or(np.logical_and(np.logical_and
                                (df.is_inc_v == 1  , df.delinq_2yrs >= 1),
                                pd.to_numeric(df.loan_status) >= 5), 
                                  pd.to_numeric(df.grade)>=4).astype('uint8'))


np.logical_or(np.logical_and(np.logical_and(df.is_inc_v == 1  , df.delinq_2yrs >= 1) , 
               pd.to_numeric(df.loan_status) >= 5) , pd.to_numeric(df.grade)>=4).value_counts()

# Creating a new variable"red_flag" which takes into account if the income has 
# NOT been verified AND if there was at least one delenquincy the past 2 years 
# AND if the loan status is a 4 or higher.

# Red flag is the variable for the company to take a good look at on whether or
# not they should give the client the loan since their income is not verified
# they had at least 1 delinquincy in the past two years and the loan status is
# a 5 (Late(31-120 days)) or higher. 
# OR is their loan grade an E or higher? 

# There are 19896 people that based on my criteria are red flags for loans.


# In[39]:


# Descriptive Statistics

df.installment.describe().apply(lambda x: format(x, 'f'))

# Here we can see there are at least two outliers since the mean is 446 and
# the min is 23 while the max is 1409.99


# In[40]:


# !pip install matplotlib.pyplot
# !pip install seaborn

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("ggplot")
import seaborn as sns


plt.hist(df.installment, color = "forestgreen")
plt.title("Histogram for Installments")
plt.xlabel("Installments")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

sns.boxplot(df.installment)
plt.title("Boxplot for Installments")
plt.tight_layout()
plt.show()

# When we look at our histogram we can see it is skewed to the right
# For our boxplot we can see we have a numerous amount of outliers
# I believe keeping the outliers is important in this dataset because
# there are valid reasons to why there are outliers, the installment
# is based on the loan amount which is also based on annual income
# some people make more money than others thus they have a larger installment number.


# In[41]:


df["monthly_loan_income_%"].describe().apply(lambda x: format(x, 'f'))


# In[42]:


plt.hist(df['monthly_loan_income_%'], color = "forestgreen")
plt.title("Histogram for installment:monthly income percentage")
plt.xlabel("Percentage Amount")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

sns.boxplot(df["monthly_loan_income_%"])
plt.title("Boxplot for installment:monthly income percentage")
plt.tight_layout()
plt.show()

# Here we can see again our histogramis skewed to the right and
# there are outliers for the loan amount


# In[43]:


cols = list(df)
cols.insert(-1, cols.pop(cols.index("term")))
cols.insert(-2, cols.pop(cols.index("grade")))
cols.insert(-3, cols.pop(cols.index("home_ownership")))
cols.insert(-4, cols.pop(cols.index("is_inc_v")))
cols.insert(-5, cols.pop(cols.index("loan_status")))
cols.insert(-6, cols.pop(cols.index("pymnt_plan")))
cols.insert(-7, cols.pop(cols.index("inq_last_6mths")))
cols.insert(-8, cols.pop(cols.index("initial_list_status")))
df = df.ix[:, cols]

# Rearranging the columns so we can slice it easier for our scaling


# In[44]:


#!pip install scipy
from scipy.stats import skew

print("Mean of skewness:",np.round(np.mean(skew(df.iloc[:-1,0:26])),2))
print("Median of skewness:",np.round(np.median(skew(df.iloc[:-1,0:26])),2))
print("Min of skewness:",np.round(min(skew(df.iloc[:-1,0:26])),2))
print("Max of skewness:",np.round(max(skew(df.iloc[:-1,0:26])),2))

# Here we can see there is a huge skewness of the data so we can use a scaler
# to help with the skewness


# In[45]:


from sklearn.preprocessing import scale

df.iloc[:-1,0:26] = scale(df.iloc[:-1,0:26])


# In[46]:


# So now our business problem, can we create a model that can give us a good
# accuracy on who would be a potential risk when applying for a loan?


# In[47]:


df.potential_risk.value_counts()

# Here we can see there is in imbalancement on the target variable so we can
# resample it to rebalance it


# In[48]:


from sklearn.utils import resample

# Separate majority and minority classes
df_majority = df[df.potential_risk==0]
df_minority = df[df.potential_risk==1]
 
# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=102028,    # to match majority class
                                 random_state=123) # reproducible results
 
# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
 
# Display new class counts
df_upsampled.potential_risk.value_counts()


# In[49]:


from sklearn.model_selection import train_test_split

X_risk = df_upsampled.drop(["potential_risk" , "red_flag"] , axis = 1)

# Dropping the red_flag column because it is similar to our target variable:
# potential risk

y_risk = df_upsampled.potential_risk


X_train, X_val, y_train, y_val = train_test_split(X_risk, y_risk, 
                                                  test_size = 0.25, random_state=42)


# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


logreg = LogisticRegression()
logreg.fit(X_train , y_train)
y_pred = logreg.predict(X_val)

print(confusion_matrix(y_val,y_pred))
print(classification_report(y_val , y_pred))
print(accuracy_score(y_val , y_pred))
print("AUROC:", roc_auc_score(y_val.tolist(), y_pred))
print("This is the category the model is predicting:",np.unique(y_pred))

# Our logistic regression accuracy gives us an ~92% along with our AUROC score
# which is solid.


# In[51]:


from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()

tree.fit(X_train , y_train.tolist())
y_pred_DT = tree.predict(X_val)
print(classification_report(y_val , y_pred_DT))
print("accuracy:",round(accuracy_score(y_val , y_pred_DT)*100,2) , "%")
print("AUROC:", roc_auc_score(y_val.tolist(), y_pred_DT))
print("This is the category the model is predicting:",np.unique(y_pred))


# In[52]:


from sklearn.ensemble import RandomForestClassifier

RF=RandomForestClassifier(n_estimators=100)

RF.fit(X_train,y_train.tolist())

y_pred_RF=RF.predict(X_val)
print(classification_report(y_val , y_pred_RF))
print("accuracy:",(accuracy_score(y_val , y_pred_RF)*100) , "%")
print("AUROC:", roc_auc_score(y_val.tolist(), y_pred_RF))
# why AUROC isnt effective, AUROC will not be effective for business decisions like false positives, false negatives
print("This is the category the model is predicting:",np.unique(y_pred_RF))

