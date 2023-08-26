##importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV



from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier



##import file 
insurance = pd.read_csv("C:/Users/USER/OneDrive/Masaüstü/insurance.csv")
df = insurance.copy()



df.head()  #to observe first 5 data
df.shape   #to observe shape of data
df.info()    #to obtain general information
df.describe() #to obtain summary statistics
df.isnull().sum() #to observe null values
df.isna().sum()  #to observe missing values
df.size  #to observe size
df.columns #to observe columns labels
df.nunique()  #to observe nonunique variables
df.duplicated().sum()  #to observe duplicated values



df["sex"].value_counts()   #counts the values of sex column
df["sex"].value_counts(normalize = True)   #fractional values of sex column

df["children"].value_counts()   #counts the values of children column
df["children"].value_counts(normalize = True)   #fractional values  of children column

df["smoker"].value_counts()    #counts the values of smoker column
df["smoker"].value_counts(normalize = True)    #fractional values  of smoker column

df["region"].value_counts()    #counts the values of region column
df["region"].value_counts(normalize = True)    #fractional values  of region column



age_categories = ["0-20","20-40","40-60","60+"]
df["age_groups"] = pd.cut(df["age"], bins=[0, 20, 40, 60, 80], labels = age_categories, right = False)


bmi_categories = ["0-20","20-40","40-60"]
df["bmi_groups"] = pd.cut(df["bmi"], bins=[0, 20, 40, 60], labels = bmi_categories, right = False)



#observe the distributions of selected columns
sns.distplot(df.charges)
plt.show()

sns.distplot(df.bmi)
plt.show()

sns.distplot(df.age)
plt.show()


#sns.heatmap ile korelasyon incelemesi
sns.heatmap(data = df.corr(), annot = True)
print(df.corr()['charges'])




##Examine the distribution of Bmi (Body Mass Index)
sns.histplot(data = df, x = df["bmi"])
plt.title("Distribution of BMI")
plt.show()


sns.kdeplot(data = df, x = df["bmi"])
plt.title("Density Curve for BMI")
plt.show()


sns.boxplot(data = df, x = df["bmi"])
plt.title("Boxplot for BMI")
plt.show()



##Examine the relationship between “smoker” and “charges”
sns.catplot(data = df, x = "smoker", y = "charges", kind = "bar")
plt.xlabel("Smoker answers")
plt.ylabel("Charges numbers")
plt.title("Smoker vs Charges Plot")
plt.show()




##Examine the relationship between “smoker” and “region”
sns.countplot(data = df, x = "smoker", hue = "region")
plt.xlabel("Smoker")
plt.title("Region Plot")
plt.show()


sns.countplot(data = df, x = "region", hue = "smoker")
plt.xlabel("Region")
plt.title("Region Plot")
plt.show()


##Examine the relationship between “bmi” and “sex”.
sns.catplot(data = df, x = "sex", y = "bmi", kind = "box")
plt.xlabel("Sex")
plt.ylabel("BMI")
plt.title("Sex vs BMI Boxlot")
plt.show()


sns.catplot(data = df, x = "sex", y = "bmi", kind = "violin")
plt.xlabel("Sex")
plt.ylabel("BMI")
plt.title("Sex vs BMI Violinplot")
plt.show()




##Find the "region" with the most "children"
sns.barplot(data = df, x = "region", y = "children")
plt.xlabel("Region")
plt.ylabel("Children")
plt.title("Region vs Children")
plt.show()



##Examine the relationship between “age” and “bmi”.
sns.kdeplot(data = df, x = "bmi", hue = "age_groups")
plt.xlabel("BMI")
plt.title("BMI Density Plot")
plt.show()


sns.catplot(data = df, x = "bmi", y = "age_groups", kind = "box")
plt.xlabel("BMI")
plt.ylabel("Age Groups")
plt.title("BMI Boxplot")
plt.show()


sns.catplot(data = df, x = "bmi", y = "age_groups", kind = "bar")
plt.xlabel("BMI")
plt.ylabel("Age Groups")
plt.title("BMI Barplot")
plt.show()


sns.scatterplot(data = df, x = "age", y = "bmi", hue = "bmi_groups")
plt.xlabel("BMI")
plt.ylabel("Age Groups")
plt.title("BMI Barplot")
plt.show()



##Examine the relationship between “bmi” and “children”
sns.barplot(data = df, x = "children", y = "bmi")
plt.show()

sns.boxplot(data = df, x = "children", y = "bmi")
plt.show()


##Is there an outlier in the "bmi" variable? Please review.
sns.boxplot(data = df, x = "bmi")
plt.show()



##Examine the relationship between “bmi” and “charges”
sns.scatterplot(data = df, x = "charges", y = "bmi")
plt.xlabel("Charges")
plt.ylabel("BMI")
plt.title("Charges vs BMI Plot")
plt.show()


plt.scatter(data = df, x = "charges", y = "bmi")
plt.xlabel("Charges")
plt.ylabel("BMI")
plt.title("Charges vs BMI Plot")
plt.show()


##Examine the relationship between “region”, “smoker” and “bmi” using bar plot.
sns.catplot(data = df, x = "region", y = "bmi", hue = "smoker", kind = "bar")
plt.xlabel("Smoker")
plt.ylabel("BMI")
plt.title("Smoker vs BMI Plot with region")
plt.show()


sns.catplot(data = df, x = "smoker", y = "bmi", hue = "region", kind = "bar")
plt.xlabel("Smoker")
plt.ylabel("BMI")
plt.title("Smoker vs BMI Plot with region")
plt.show()




##Data Preprocessing
##In this section, prepare the data you have, for training the model.
numerical_columns = ["age","bmi","children","charges"]
categorical_columns = ["sex", "smoker", "region","age_groups","bmi_groups"]


#applying one-hot encoding on categorical columns
data_one_hot_encoded = pd.get_dummies(data = df, columns = categorical_columns, 
                              prefix = categorical_columns,
                              drop_first = True)

data_one_hot_encoded.head()



#appling label encoder on categorical columns
label_encoder = LabelEncoder()


#inspect sex
label_encoder.fit(df.sex.drop_duplicates()) 
df.sex = label_encoder.transform(df.sex)
df.sex


# smoker or not
label_encoder.fit(df.smoker.drop_duplicates()) 
df.smoker = label_encoder.transform(df.smoker)
df.smoker

#region
label_encoder.fit(df.region.drop_duplicates()) 
df.region = label_encoder.transform(df.region)
df.smoker



##Split your dataset into X_train,X_test, y_train, y_test
X = df.drop(columns = ["charges","age_groups","bmi_groups"], axis = 1)
y = df["charges"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)



linear_model = LinearRegression()
linear_model_fit = linear_model.fit(X_train, y_train)
predicted_lm = linear_model.predict(X_test)
plt.scatter(y_test, predicted_lm)
plt.plot(y_test,y_test,"r")


print(f"Mean absolute error (MAE): {mean_absolute_error(y_test,predicted_lm)}")
print(f"Mean squared error (MSE): {mean_squared_error(y_test,predicted_lm)}")
print(f"Root Mean Squared error (RMSE): {np.sqrt(mean_squared_error(y_test,predicted_lm))}")
print(f"R-squared: {r2_score(y_test,predicted_lm)}")




#Ridge Model
ridge_model = Ridge()
ridge_model_fit = ridge_model.fit(X_train, y_train)
predicted_ridge = ridge_model.predict(X_test)


print(f"Mean absolute error (MAE): {mean_absolute_error(y_test,predicted_ridge)}")
print(f"Mean squared error (MSE): {mean_squared_error(y_test,predicted_ridge)}")
print(f"Root Mean Squared error (RMSE): {np.sqrt(mean_squared_error(y_test,predicted_ridge))}")
print(f"R-squared: {r2_score(y_test,predicted_ridge)}")


