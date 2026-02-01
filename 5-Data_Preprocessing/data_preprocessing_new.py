import pandas as pd
import numpy as np
#let us import dataset
df=pd.read_csv("ethnic diversity.csv")
#cheak data types of columns
df.dtypes
#salaries data type is float,let us convert into int
df.Salaries=df.Salaries.astype(int)
df.dtypes
#convert age data type it must be float
df.age=df.age.astype(float)
df.dtypes
#identify the duplicates
df_new=pd.read_csv("education.csv")
duplicate=df_new.duplicated()
duplicate
sum(duplicate)
# new dataset
df_new1=pd.read_csv("mtcars_dup.csv")
duplicate1=df_new1.duplicated()
duplicate1
sum(duplicate1)
#remove duplicates using drop
df_new2=df_new1.drop_duplicates()
duplicate2=df_new2.duplicated()
duplicate2
sum(duplicate2)


#outliers treatment
import pandas as pd
import seaborn as sns
df=pd.read_csv("ethnic diversity.csv")
#find outliers in salaries
sns.boxplot(df.Salaries)
#there are outliers
#let us cheak otliers in age column
sns.boxplot(df.age)
#let us calculate iqr
IQR = df.Salaries.quantile(0.75)-df.Salaries.quantile(0.25)
IQR
#lower-upper limited
lower_limit=df.Salaries.quantile(0.25)-1.5*IQR
upper_limit=df.Salaries.quantile(0.75)+1.5*IQR
#trimming
import numpy as np
outliers_df=np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))
df_trimmed=df.loc[~outliers_df]
df.shape
#(310,13)
df_trimmed.shape
#(306,13)



###########################
#replacement Technique
# drawback of trimming technique we are losing data
import pandas as pd
import seaborn as sns
import numpy as np
df=pd.read_csv("ethnic diversity.csv")
df_replaced=pd.DataFrame(np.where(df.Salaries>upper_limit,upper_limit,np.where(df.Salaries,lower_limit,df.Salaries)))
sns.boxplot(df_replaced[0])#column name,ref df_replaced


##########################################3
#winsorizer
from feature_engine.outliers import Winsorizer
import seaborn as sns
winsor=Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = ['Salaries'])
df_t = winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])

###########################################
import pandas as pd
df=pd.read_csv("ethnic diversity.csv")
numeric_df=df.select_dtypes(include="number")
numeric_df.var()
#empid and zip are nominal data
#salaries and age are having considerable variance
#1.slect only numeric column
numeric_df=df.select_dtypes(include="number")
#2.find variance of each numeric column
variances=numeric_df.var()
#3.identify columns with zero variance
zero_val_cols=variances[variances==0].index.tolist()
#4.drop these columns from original dataframe
df_cleaned=df.drop(columns=zero_val_cols)
#print the columns droped
print("dropped columns with zero variance:",zero_val_cols)



#############################################
#mean and median imputer are used for numerical data
#mode is used for nomianl or categorical data
#mean imputer
import pandas as pd
import numpy as np
df=pd.read_csv("ethnic diversity.csv")
from sklearn.impute import SimpleImputer
mean_imputer=SimpleImputer(missing_values=np.nan,strategy="mean")
#cheack the dataframe
df["Salaries"]=pd.DataFrame(mean_imputer.fit_transform(df[["Salaries"]]))
#cheack the dataframe
df["Salaries"].isna().sum()
#0

#median imputer
median_imputer=SimpleImputer(missing_values=np.nan,strategy="median")
#cheack the dataframe
df["age"]=pd.DataFrame(median_imputer.fit_transform(df[["age"]]))
#cheack the dataframe
df["age"].isna().sum()
#0
#####################################
#mode imputer
mode_imputer=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
#cheack the dataframe
df["Sex"]=pd.DataFrame(mode_imputer.fit_transform(df[["Sex"]]))
df["Sex"].isna().sum()
#0
df["MaritialDesc"]=pd.DataFrame(mode_imputer.fit_transform(df[["MaritialDesc"]]))
df["MaritialDesc"].isna().sum()
#0









