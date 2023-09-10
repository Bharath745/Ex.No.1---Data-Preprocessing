# Ex.No.1---Data-Preprocessing
## AIM:
To perform Data preprocessing in a data set downloaded from Kaggle
## REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook
## RELATED THEORETICAL CONCEPT:
Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:
Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :
For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
1. Importing the libraries
2. Importing the dataset
3. Taking care of missing data
4. Encoding categorical data
5. Normalizing the data
6. Splitting the data into test and train

## PROGRAM:
```python
import pandas as pd
df=pd.read_csv("/content/Churn_Modelling.csv")
df.head()
df.isnull().sum()
df.drop(["RowNumber","Age","Gender","Geography","Surname"],inplace=True,axis=1)
print(df)
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(x)
print(y)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df1 = pd.DataFrame(scaler.fit_transform(df))
print(df1)
from sklearn.model_selection import train_test_split
xtrain,ytrain,xtest,ytest=train_test_split(x,y,test_size=0.2,random_state=2)
print(xtrain)
print(len(xtrain))
print(xtest)
print(len(xtest))
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df1 = sc.fit_transform(df)
print(df1)
```

## OUTPUT:
![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/2a4b7331-c125-48cd-bcec-64b69f8ef6b6)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/63af323d-c7ac-481d-9bb8-c4ec01475ea3)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/5d61992f-13a6-49bf-91e0-cff8c4fc1d5a)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/d2ce4de4-b03e-4bcc-9199-455173050bef)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/05865699-6ffa-49e9-a63b-1c0cbf1e2229)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/69489695-a9a2-4238-addf-3d1a7087eee7)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/7d332768-12d7-4567-9fda-3576c9679cf0)

![image](https://github.com/Bharath745/Ex.No.1---Data-Preprocessing/assets/94508354/27714630-79a0-445a-934d-20cd0191b0ca)


## RESULT
The Data preprocessing is performed over a data set successfully.
