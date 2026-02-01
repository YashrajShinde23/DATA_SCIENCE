#create pandas dataframe from list
import pandas as pd
technologies=[["spark",20000,"30days"],
              ["pandas",20000,"40days"]]
df=pd.Dataframe(technologies)
print(df)
#add coloumn name and row name
column_names=["courses","fees","duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,column=column_names,index=row_label)
print(df)



#create dataframe from dictionary
import pandas as pd
technologies={
    "courses":["spark","pyspark","hadoop"],
    "fees":[20000,25000,26000],
    "duration":["30days","40days","35days"],
    "discount":[1000,2300,1500]
    }
df=pd.DataFrame(technologies)
df


#convert dataframe to csv
df.to_csv("data_file.csv")








#
import numpy as np
import pandas as pd
technologies={
    "courses":["spark","pyspark","hadoop","python","pandas",None,"python","spark"],
    "fees":[22000,25000,23000,24000,np.nan,25000,25000,22000],
    "duration":["30days","50days","55days","40days","60days","35days","","50days"],
    "discount":[1000,2300,1000,3000,5000,1200,1300,1400]}
row_labels=["r0","r1","r2","r3","r4","r5","r6","r7"]
df=pd.DataFrame(technologies,index=row_labels)
print(df)

#dataframe properties
df.shape #(8,4)
df.size #32
df.columns

df.columns.values
df.index
df.dtypes


#accessing one column content
df["fees"]
#accessing two column content
df[["fees","duration"]]

#select certain rows 
df2=df[6:]
df2

#accessing certain cell from column "duration"
df["duration"][3]

#subtracting specific values from column
df["fees"]=df["fees"]-500
df["fees"]

#describe dataframe
#describe datframe for all numeric column
df.describe()


#rename()-rename pandas dataframe columns
df=pd.DataFrame(technologies,index=row_labels)
#assign new header by setting new column names
df.columns=["a","b","c","d"]
df

#rename column using rename() method
df=pd.DataFrame(technologies,index=row_labels)
df.columns=["A","B","C","D"]
df2=df.rename({"A":"c1","B":"c2"},axis=1)
df2=df.rename({"C":"c3","D":"c4"},axis="columns")
df2=df.rename(columns={"A":"c1","B":"c2"})
df=pd.DataFrame(technologies,index=row_labels)
df


#drop dataframe rows and column
df1=df.drop(["r1","r2"])
df1
df=pd.DataFrame(technologies,index=row_labels)
#delete rows by position
df1=df.drop(df.index[[1,3]])
#delete rows by index range
df1=df.drop(df.index[2:])


#when you have default index for rows
df=pd.DataFrame(technologies)
df1=df.drop(0)
df=pd.DataFrame(technologies)
df1=df.drop([0,3])#it will delete row0 and row3
df1=df.drop(range(0,2))#it will delete 0 and 1


############################################
#select rows by index
df=pd.DataFrame(technologies,index=row_labels)
df2=df.iloc[2]#slect row by index
df2=df.iloc[[2,3,6]]#select rows by index list
df2=df.iloc[1:5]#select rows by integer range
df2=df.iloc[:1]#select first row
df2=df.iloc[:3]#select first 3 rows
df2=df.iloc[-1:]#select last row
df2=df.iloc[-3:]#select last 3 row
df2=df.iloc[::2]#select alternate rows



#select rows by index names
df2=df.loc["r2"]#select row by index label
df2=df.loc[["r2","r3","r6"]]#select row by index label
df2=df.loc["r1":"r5"]#select rows by index range
df2=df.loc["r1":"r5":2]#select alternate rows


#slect columns by name or index
df2=df["courses"]
#select multiple columns
df2=df[["courses","fees","duration"]]

#using loc to take column slices
#select multiple columns
df2=df.loc[:,["courses","fees","duration"]]
#select random column
df2=df.loc[:,["courses","fees","discount"]]
#select columns between two columns
df2=df.loc[:,"fees":"discount"]
#select column by range
df2=df.loc[:,"duration":]
#all columns upto duration
df2=df.loc[:,:"duration"]
#slect every alternate column
df2=df.loc[:,::2]



#drop rows that has nan/none/null values
#delete rows that has nan/none/null values
import pandas as pd
import numpy as np
technologies={
    "courses":["spark","pyspark","hadoop","python"],
    "fee":[20000,25000,26000,22000],
    "duration":["","40days",np.nan,None],
    "discount":[1000,2300,1500,1200]
    }
indexes=["r1","r2","r3","r4"]
df=pd.DataFrame(technologies,index=indexes)
print(df)
df=pd.DataFrame(technologies,index=indexes)
df2=df.dropna()
print(df2)
#none type means no value/empty
#np.nan float missing or invalid number


#first drop rows with nan(np.nan and none)
df_clean=df.dropna(subset=["duration"])
#then drop rows with empty strings
df_clean=df_clean[df_clean["duration"]!=""]
print(df_clean)


#change all columns to same type in pandas
df=df.astype(str)
print(df.dtypes)
# change type for one or multiple columns in pandas
df=df.astype({"fee":int,"discount":float})
print(df.dtypes)


#convert data type for columns in list
df=pd.DataFrame(technologies)
cols=["fee","discount"]
df[cols]=df[cols].astype(float)
df.dtypes
#by using loop
for col in ["fee","discount"]:
    df[col]=df[col].astype(float)
    
    
#raise or ignore error when convert column type fails
df=df.astype({"courses":int},errors="ignore")
df.dtypes
#generates errors
df=df.astype({"courses":int},errors="raise")

#using dataframe.to_numeric() to convert numeric types
#converts feed column to numeric type
df["fee"]=pd.to_numeric(df["fee"])
print(df.dtypes)

#convert multiple numeric types using apply() method
#convert fee and discount to numeric types
df=pd.DataFrame(technologies)
df.dtypes
df[["fee","discount"]]=df[["fee","discount"]].apply(pd.to_numeric)
print(df.dtypes)


#example of get the number of rows in dataframe
rows_count=len(df.index)
rows_count
rows_count=len(df.axes[0])
rows_count
#######################################
df=pd.DataFrame(technologies)
row_count=df.shape[0]#returns number of rows
col_count=df.shape[1]#return number of column
print(row_count)
print(col_count)


#using dataframe.apply() to apply fucntion add column
import pandas as pd
import numpy as np
data=[(3,5,7),(2,4,6),(5,8,9)]
df=pd.DataFrame(data,columns=["A","B","C"])
print(df)


def add_3(x):
    return x+3
df2=df.apply(add_3)
print(df2)

#########################
#using apply function single column
df=pd.DataFrame(data,columns=["A","B","C"])
def add_4(x):
    return x+4
df["B"]=df["B"].apply(add_4)
df["B"]
#apply to multiple column
df=pd.DataFrame(data,columns=["A","B","C"])
df[["A","B"]]=df[["A","B"]].apply(add_3)

#apply a lambda function to each column
df2=df.apply(lambda x:x+10)
df2

#apply lambda to single column
df["A"]=df["A"].apply(lambda x:x-2)
print(df)


#using pandas.dataframe.transform gives same result
#as aplly as shown above
def add_2(x):
    return x+2
df=df.transform(add_2)
print(df)

#using pandas.dataframe.map() to single column
df["A"]=df["A"].map(lambda A:A/2)
print(df)


#using numpy function on single column
df["A"]=df["A"].apply(np.square)
print(df)



#
import pandas as pd
import numpy as np
data=[(3,5,7),(2,4,6),(5,8,9)]
df=pd.DataFrame(data,columns=["A","B","C"])
print(df)
#using numpy.square()method
#using numpy.square()
df["A"]=np.square(df["A"])
print(df)



#pandas groupby() with examples
import pandas as pd
import numpy as np
technologies={
    "courses":["spark","pyspark","hadoop","python","pandas","hadoop","spark","python","NA"],
    "fee":[22000,25000,23000,24000,26000,25000,25000,22000,1500],
    "duration":["30days","50days","55days","40days","60days","35days","30days","50days","40days"],
    "discount":[1000,2300,1000,1200,2500,None,1400,1600,0]
    }
df=pd.DataFrame(technologies)
print(df)
#use groupby() to compute the sum
df2=df.groupby(["courses"]).sum()
print(df2)

#groupby multiple columns
df2=df.groupby(["courses","duration"]).sum().reset_index()
print(df2)

#add index to grouped data
df2=df.groupby({"courses","duration"})


#
import pandas as pd
import numpy as np
technologies={
    "courses":["spark","pyspark","hadoop","python","pandas"],
    "fee":[22000,25000,23000,24000,26000],
    "duration":["30days","50days","55days","40days","60days"],
    "discount":[1000,2300,1000,1200,2500]
    }
df=pd.DataFrame(technologies)
print(df)
#get list of all column names from headers
column_headers=list(df.columns)
column_headers
#another way to get column names
column_headers=list(df)
column_headers

##########################################
#pandas shuffle dataframe
import pandas as pd
import numpy as np
technologies={
    "courses":["spark","pyspark","hadoop","python","pandas"],
    "fee":[22000,25000,26000,22000,24000],
    "duration":["30days","50days","55days","40days","60days"],
    "discount":[1000,2300,1000,1200,2500]
    }
df=pd.DataFrame(technologies)
print(df)
#pandas shuffle dataframe rows
#shuffle the dataframe rows and return all rows
df1=df.sample(frac=1)
df1

#create a new index starting from zero reset the df1
df1=df.sample(frac=1).reset_index()
df1

#drop shuffle index another way to reset shuffled index
df1=df.sample(frac=1).reset_index(drop=True)
print(df1)




