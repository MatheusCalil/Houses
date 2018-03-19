import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import explained_variance_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

train = pd.read_csv('C:\\Users\\mcfal\\Desktop\\Outros\\Datasets\\Houses\\train.csv')
test = pd.read_csv('C:\\Users\\mcfal\\Desktop\\Outros\\Datasets\\Houses\\test.csv')

#checking colnames
train.columns
test.columns
len(test.columns) == len(train.columns) #false, test set do not have the SalePrice
#we will use only train set and split it

train.shape #(1460,81)

train.dtypes #checking types

#sumarizing na
na = 100*train.isnull().sum()/train.shape[0]

#removing columns with more than 50% of NA
train = train.drop(na[na>50].index, axis=1)

#checking for the remained columns with na
na = 100*train.isnull().sum()/train.shape[0]
na[na!=0]
#FireplaceQu have 47.2%, lets remove it
train = train.drop('FireplaceQu', axis=1)

#the others we need to fill. Let's fill with the most common value.
na = 100*train.isnull().sum()/train.shape[0]
na = na[na!=0]

for index in na.index:
    train = train.fillna({index:train[index].value_counts().index[0]})

na = 100*train.isnull().sum()/train.shape[0]

#we need to convert categorical to numeric
col = train.select_dtypes(['object']).columns
for c in col:
    lb_make = LabelEncoder()
    train[c] = lb_make.fit_transform(train[c])


#checking types
train.dtypes[train.dtypes != 'int64']

#now we can start the analysis
train.describe()

#removing the variable to predict SalePrice and store in other array
salePrice = train['SalePrice']
train = train.drop('SalePrice', axis=1)

#creating train and test variables
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    salePrice,
                                                    random_state = 42,
                                                    test_size=0.4)
#LASSO regression
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train, y_train)
explained_variance_score(y_test, reg.predict(X_test)) #0.81 is good
mean_absolute_error(y_test, reg.predict(X_test)) #22712

#Using LASSO CV and optimizing alphas
alphas = [1, 0.1, 0.001, 0.0005]
reg = linear_model.LassoCV(alphas  = alphas)
reg.fit(X_train, y_train)
explained_variance_score(y_test, reg.predict(X_test)) #0.81 is good
mean_absolute_error(y_test, reg.predict(X_test)) #22712

#checking coeficients of LASSO CV
coef = pd.Series(reg.coef_, index = train.columns)

#Most important coef: 10 with highest and 10 with lowest
imp_coef = pd.concat([coef.sort_values().head(10),
                     coef.sort_values().tail(10)])

#RIDGE
reg = linear_model.Ridge (alpha = .5)
reg.fit(X_train, y_train)
explained_variance_score(y_test, reg.predict(X_test)) #0.81 is good
mean_absolute_error(y_test, reg.predict(X_test)) #22612

#feature selection
trainSelectedFeatures = SelectKBest(f_regression, k=20).fit_transform(train,salePrice)
X_train, X_test, y_train, y_test = train_test_split(trainSelectedFeatures,
                                                    salePrice,
                                                    random_state = 42,
                                                    test_size=0.4)
#LASSO regression
reg = linear_model.Lasso(alpha = 0.1)
reg.fit(X_train, y_train)
explained_variance_score(y_test, reg.predict(X_test)) #0.82 is good
mean_absolute_error(y_test, reg.predict(X_test)) #23207 
