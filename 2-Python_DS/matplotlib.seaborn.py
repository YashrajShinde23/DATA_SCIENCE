#
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
sns.displot(tips.total_bill,kde=True)

#
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
sns.displot(tips.tip,kde=True)

#
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
sns.displot(tips.size,kde=True)

#
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset("tips")
tips.head()
sns.jointplot(x=tips.tip,y=tips.total_bill)

#
sns.jointplot(x=tips.tip,y=tips.total_bill,kind="reg")
sns.jointplot(x=tips.tip,y=tips.total_bill,kind="hex")
sns.pairplot(tips,kind="reg")

#
tips.time.value_counts()
sns.pairplot(tips,hue="time")

#
sns.pairplot(tips,hue="day")

#
sns.heatmap(tips.corr(numeric_only=True),annot=True)

#
sns.boxplot(tips.total_bill)
#their are outliers in total_bill
sns.boxplot(tips.tip)
#their are outliers in total_bill
sns.countplot(tips.day)
#there are more number of people on saturday
#then on sunday and moderate number of people on thursday
#and few number of people on friday
sns.countplot(tips.sex)

#there are more number of male coustumer
#there are less number of female coustumer
tips.sex.value_counts().plot(kind="pie")
tips.sex.value_counts().plot(kind="bar")

sns.countplot(data=tips[tips.time=="Dinner"],x="day")
#there are more number of people on saturday
sns.countplot(data=tips[tips.time=="Lunch"],x="day")
#there are more number of people on thursday

#this is facetgrid of histogram showing totaol distribution
#of total bills across different smoking status and time of day
fg=sns.FacetGrid(tips,row="smoker",col="time")
fg.map(sns.histplot,"total_bill")


#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Mall_customers.csv") 
df.columns=["cust_id","Genre","Age","Ann_income","Spend_scor"]
sns.displot(df.Age,kde=True)
sns.displot(df.Ann_income,kde=True)
sns.displot(df.Spend_scor,kde=True)
df.dtypes
sns.jointplot(x=df.Age,y=df.Spend_scor)
#maximum spending occurs during 20-30 ,even at 50 too
sns.jointplot(x=df.Age,y=df.Spend_scor,kind="reg")
#spending reduces as age increases
sns.jointplot(x=df.Age,y=df.Spend_scor,kind="hex")
#each spends at age of 30 and 50








#matplotlib starts here
#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#load dataset
tips=sns.load_dataset("tips")

#1.histogram of total bill
plt.hist(tips["total_bill"],bins=20,color="purple",edgecolor="black")
plt.title("total bill distribution")
plt.xlabel("total bill")
plt.ylabel("count")
plt.show()


#2.histogram for tip
plt.hist(tips["tip"],bins=20,color="lightgreen",edgecolor="black")
plt.title("tip distribution")
plt.xlabel("tip")
plt.ylabel("count")
plt.show()


#3.scatter plot of tip vs total bill
plt.scatter(tips["tip"],tips["total_bill"],color="purple",alpha=0.6)
plt.title("tip vs total bill")
plt.xlabel("tip")
plt.ylabel("total bill")
plt.show()


#simple coorelation heatmap
corr=tips.corr(numeric_only=True)
plt.imshow(corr,cmap="coolwarm",interpolation="none")
plt.colorbar()
plt.xticks(range(len(corr)),corr.columns,rotation=45)
#(len(corr)) generates the postion for the ticks
plt.ysticks(range(len(corr)),corr.columns)
plt.title("correlation matrix")
plt.show()


#5.box plot
plt.boxplot(tips["total_bill"])
plt.title("boxplot - total_bill")
plt.show()

plt.boxplot(tips["tip"])
plt.title("boxplot - tip")
plt.show()

#6.bar graph of days
tips["day"].value_counts().plot(kind="bar",color="orange")
plt.title("count by days")
plt.xlabel("day")
plt.ylabel("count")
plt.show()


#7.bar graph of  gender
tips["sex"].value_counts().plot(kind="bar",color="lightblue")
plt.title("count by gender")
plt.xlabel("sex")
plt.ylabel("count")
plt.show()


#8.gender pie chart
tips["sex"].value_counts().plot(kind="pie",autopct="%1.1f%%")
#autopct=%1.1f%% is function used to display percentage value on each
#slice of pie chart


#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Salary_Data.csv")
df.columns=["age","gender","education","job","exp",""]

#histograms
plt.hist(df.Age,bins=10,edgecolor="black",alpha=0.7)
plt.title("age distribution")
plt.xlabel("age")
plt.ylabel("frequency")
plt.show()