import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings # Ignores any warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import Imputer
from scipy.stats import mode
from sklearn_pandas import CategoricalImputer

train= pd.read_csv("data/Train.csv")
test= pd.read_csv("data/Test.csv")

#checking for duplicates
idsUnique = len(set(train.Item_Identifier))
idsTotal = train.shape[0]
print(idsTotal-idsUnique)

#Univariate analysis
#Distribution of the target variable

plt.style.use('fivethirtyeight')
plt.figure(figsize=(12,7))
sns.distplot(train.Item_Outlet_Sales,bins=25)
plt.ticklabel_format(style='pltrain.head()ain',axis='x',scilimits=(0,1))
plt.xlabel("Item_Outlet_Sales")
plt.ylabel("Number Of Sales")
plt.title("Item_Outlet_Sales Distribution")

#skewness and Kurtosis
print("Skew is:",train.Item_Outlet_Sales.skew())
print("Kurtosis: %f" % train.Item_Outlet_Sales.kurt())
 
#for considering our Predictors calculating numerical feaatures
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes

#Correlation between Numericals_predictors and Target Variable
corr= numeric_features.corr()
corr.head() 

#correlation matrix
f,ax = plt.subplots(figsize=(12,9))
sns.heatmap(corr,vmax=.8,square=True);

#categorical Predictors
#Distribution of Item_Fat_Content
sns.countplot(train.Item_Fat_Content)

#Distribution of the variable Item_Type
sns.countplot(train.Item_Type)
plt.xticks(rotation=0)

#Distribution of Outlet_Size
sns.countplot(train.Outlet_Size)

#Distribution of the variable Outlet_Location_Type
sns.countplot(train.Outlet_Location_Type)

#Distribution of Outlet_Type
sns.countplot(train.Outlet_Type)
plt.xticks(rotation=90)

#Bivariate Analysis
#corr.head()
#Item_Weight and Item_Outlet_Sales analysis
plt.figure(figsize=(12,7))
plt.xlabel("Item_Weight")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Weiight and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Weight, train["Item_Outlet_Sales"], '.',alpha=0.3)

#Item_Visibility and Item_Outlet_Sales Analysis

plt.figure(figsize=(12,7))
plt.xlabel("Item_Visibility")
plt.ylabel("Item_Outlet_Sales")
plt.title("Item_Visibility and Item_Outlet_Sales Analysis")
plt.plot(train.Item_Visibility, train["Item_Outlet_Sales"], '.',alpha=0.3)

#Outlet_Establishment_Year and Item_Outlet_Sales analysis
Outlet_Establishment_Year_pivot = \
train.pivot_table(index= 'Outlet_Establishment_Year', values= "Item_Outlet_Sales", aggfunc= np.median)
Outlet_Establishment_Year_pivot.plot(kind='bar', color='blue',figsize=(12,7))
plt.xlabel("Outlet_Establishment_Year")
plt.ylabel("Sqrt Item_Outlet_Sales")
plt.title("Impact of Outlet_Establishment_Year on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

#Categorical Variables

#Impact of Item_Fat_Content on Item_Outlet_Sales
Item_Fat_Content_pivot= \
train.pivot_table(index='Item_Fat_Content',values="Item_Outlet_Sales", aggfunc=np.median)
Item_Fat_Content_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel("Item_Fat_Content")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Item_Fat_Content on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

#Impact of Outlet_Idetifier on Item_Outlet_Sales
Outlet_Identifier_pivot = \
train.pivot_table(index='Outlet_Identifier',values="Item_Outlet_Sales", aggfunc=np.median)
Outlet_Identifier_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel("Outlet_Identifier")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Identifier on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()
 
#Impact of Outlet_Size on Item_outlet_Sales
Outlet_Size_pivot= \
train.pivot_table(index='Outlet_Size' , values= "Item_Outlet_Sales",aggfunc= np.median)
Outlet_Size_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel("Outlet_Size")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Size on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

#Impact of Outlet_Type on Item_Outlet_Sales
Outlet_Type_pivot= \
train.pivot_table(index='Outlet_Type' , values= "Item_Outlet_Sales",aggfunc= np.median)
Outlet_Type_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel("Outlet_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

#Impact of Outlet_Location_Type on Item_Outlet_Sales
Outlet_Location_Type_pivot= \
train.pivot_table(index='Outlet_Location_Type' , values= "Item_Outlet_Sales",aggfunc= np.median)
Outlet_Location_Type_pivot.plot(kind='bar',color='blue',figsize=(12,7))
plt.xlabel("Outlet_Location_Type")
plt.ylabel("Item_Outlet_Sales")
plt.title("Impact of Outlet_Location_Type on Item_Outlet_Sales")
plt.xticks(rotation=0)
plt.show()

#Data Pre-Processing
#join Train and Test dataset

#Creating source column to later separate the data easily
train['source']= 'train'
test['source'] = 'test'

data= pd.concat([train,test],ignore_index=True)
print(data.shape)

#checking the percentage of null values per variable
data.isnull().sum()/data.shape[0]*100

#Imputing the mean for Item_weight missing values
item_avg_weight = data.pivot_table(values='Item_Weight',index='Item_Identifier')
print(item_avg_weight)

data[:][data['Item_Identifier']=='DRI11']
#def impute_weight(cols):
    #Weight = cols[0]
    #Identifier = cols[1]
    
    #if pd.isnull(Weight):
        #return item_avg_weight['Item_Weight']
#[item_avg_weight.index == Identifier]
    #else:
        #return Weight
print ('Orignal #missing: %d'%sum(data['Item_Weight'].isnull()))
#data['Item_Weight'] = data[['Item_Weight','Item_Identifier']].apply(impute_weight,axis=1).astype(float)
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(data[['Item_Weight']])
print(data[['Item_Weight']])
data[['Item_Weight']] = imputer.transform(data[['Item_Weight']])

print ('Final #missing: %d'%sum(data['Item_Weight'].isnull()))
       
#Imputing Outlet_Size missing values with the mode
#Determining the mode

outlet_size_mode= data.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=lambda x:x.mode())
imputer1= CategoricalImputer(missing_values='NaN',strategy='most_frequent')
imputer1 = imputer1.fit(data['Outlet_Size'])
print(data['Outlet_Size'])
data[['Outlet_Size']] = imputer1.transform(data[['Outlet_Size']])
print(data['Outlet_Size'])

#Feature Engineering
#checking whether we should combine Outlet_Type or not

data.pivot_table(values='Item_Outlet_Sales',columns='Outlet_Type')

#values are significantly different so leave these

#Considering 0 item_visibility as missing we should impute missing values for these data
print(data[['Item_Visibility']])
#print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))
data['Item_Visibility'].replace([data['Item_Visibility']==0],np.NaN)
imputer2= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer2 = imputer2.fit(data[['Item_Visibility']])
print ('Initial #zeros: %d'%sum(data['Item_Visibility'] == 'NaN'))
data[['Item_Visibility']] = imputer.transform(data[['Item_Visibility']])
print ('Final #zeros: %d'%sum(data['Item_Visibility'] == 0))

#Determining the years of Operations of a store
#the data is from 2013
data['Outlet_Years'] = 2013 - data['Outlet_Establishment_Year']
data['Outlet_Years'].describe()

#Creating a broad category of Item_Type
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})

data['Item_Type_Combined'].value_counts()

#Modifying categories of Item_fat_Content
print('Original Categories')
print(data['Item_Fat_Content'].value_counts())

print('\n')
print('Modified Categories')
data['Item_Fat_Content']= data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
print(data['Item_Fat_Content'].value_counts())

#Creating special categories for non-consumable and Fat-content
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content']= "Non-Edible"
data['Item_Fat_Content'].value_counts()

#Now applying Feature Transformations
#Creating variable Item_Visibility_Mean_Ratio
#func = lambda x: x['Item_Visibility']/visibility_item_avg['Item_Visibility'][visibility_item_avg.index == x['Item_Identifier']][0]
#data['Item_Visibility_MeanRatio']= data.apply(func,axis=1).astype(float)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#New variable for outet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])

var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']
for i in var_mod:
    data[i] = le.fit_transform(data[i])
#Dummy Variables:
data = pd.get_dummies(data, columns =['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

data.dtypes

#Exporting Data

#Droping the columns which has been converted to different types
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train
train = data.loc[data['source']=='train']
test = data.loc[data['source']=='test']

#Droping unnecessary columns
test.drop(['Item_Outlet_Sales','source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)

#Exporting files as modified versions of test and train csv files
train.to_csv("data/train_modified.csv",index=False)
test.to_csv("data/test_modified.csv",index=False)

#Now Model buliding
train_df = pd.read_csv('data/train_modified.csv')
test_df = pd.read_csv('data/test_modified.csv')

#Defining a generic function that takes the different algorithms and generate the output and perform cross_validation on it
#Defining target and ID columns
target = 'Item_Outlet_Sales'
IDcol = ['Item_Identifier','Outlet_Identifier']

from sklearn import model_selection,metrics
def modelfit(alg,dtrain,dtest,predictors,target,IDcol,filename):
    #fitting the algorithm on the data
    alg.fit(dtrain[predictors],dtrain[target])
    
    #Predicting training set
    dtrain_predictions = alg.predict(dtrain[predictors])
    
    #Since the target had been normalised
    Sq_train = (dtrain[target])**2
    
    #Performing cross-validation
    cv_score = model_selection.cross_val_score(alg,dtrain[predictors],Sq_train,cv=20,scoring='neg_mean_squared_error')
    cv_score = np.sqrt(np.abs(cv_score))
    
    #Printing model report
    print("\nModel Report")
    print("RMSE : %.4g"% np.sqrt(metrics.mean_squared_error(Sq_train.values,dtrain_predictions)))
    
    print("CV Score : Mean -%.4g | Std -%.4g | Min -%.4g | Max -%.4g" % (np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))
    
    #Predict on testing data:
    dtest[target] = alg.predict(dtest[predictors])
    
    #Export Submission file:
    IDcol.append(target)
    submission =  pd.DataFrame({ x :dtest[x] for x in IDcol})
    submission.to_csv(filename,index=False)
    
#Performing test
#Linear Regression model
from sklearn.linear_model import LinearRegression
LR = LinearRegression(normalize= True)
predictors = train_df.columns.drop(['Item_Outlet_Sales','Item_Identifier','Outlet_Identifier'])
modelfit(LR,train_df,test_df,predictors,target,IDcol,'LR.csv')

#Ridge Regression model
from sklearn.linear_model import Ridge
RR= Ridge(alpha=0.05,normalize=True)
modelfit(RR,train_df,test_df,predictors,target,IDcol,'RR.csv')

#Decision Tree Model
from sklearn.tree import DecisionTreeRegressor
DT = DecisionTreeRegressor(max_depth=15,min_samples_leaf=100)
modelfit(DT,train_df,test_df,predictors,target,IDcol,'DT.csv')

#Random Forest Model
RF= DecisionTreeRegressor(max_depth=8,min_samples_leaf=150)
modelfit(RF,train_df,test_df,predictors,target,IDcol,'RF.csv')

#The end#


    

    