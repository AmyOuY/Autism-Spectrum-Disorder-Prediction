import pandas as pd
import numpy as np
import seaborn as sns



"""read raw data into csv file with pandas format"""
newData = open("childnewData.txt", "w")
 
with open ("Autism-Child-Data.arff", "r") as f:
    for line in f:
        if len(line) != 1 and not line.startswith("@"):
            newData.write(line)
       
newData.close()   

columns = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'age', 'gender', 'ethnicity', 'jaundice', 'autism', 'country_of_res', 'used_app_before', 'result', 'age_desc', 'relation', 'class']
data = pd.read_csv("childnewData.txt", names=columns)


"""discard instances with missing values and replace abnormal age value with mean of age"""
data1 = data.replace('?', np.nan)
data1.dropna(axis = 0, inplace = True)
data1.to_csv("childDataNaAge.csv", sep = ",", index=False)


"""bar plot the counts of instances in each class"""
plotData = pd.read_csv("childDataNaAge.csv")
sns.distplot(plotData['age'], bins = 5, kde = False)
sns.countplot(x = 'class', hue = 'gender', data = plotData)
sns.set(font_scale=1.05)
sns.countplot(x = 'class', hue = 'ethnicity', data = plotData)
sns.countplot(x = 'class', hue = 'age', data = plotData)
sns.countplot(x = 'class', hue = 'jaundice', data = plotData)
sns.countplot(x = 'class', hue = 'autism', data = plotData)
sns.countplot(x = 'class', hue = 'result', palette="Set1", data = plotData)


"""change attribute values to binominal"""
data = pd.read_csv("childDataNaAge.csv")
data2 = data.replace({'gender' : 'f'}, 1).replace({'gender' : 'm'}, 0).\
replace({'jaundice' : 'yes'}, 1).replace({'jaundice' : 'no'}, 0).\
replace({'autism' : 'yes'}, 1).replace({'autism' : 'no'}, 0).\
replace({'class' : 'YES'}, 1).replace({'class' : 'NO'}, 0)


"""discar columns that show little or no information to learning"""
#data3 = data2.drop(['result'], 1)
data3 = data2.drop(['ethnicity'], 1).drop(['country_of_res'], 1).drop(['used_app_before'], 1).\
drop(['age_desc'], 1).drop(['relation'], 1).drop(['result'], 1).drop(['gender'], 1).drop(['jaundice'], 1).drop(['autism'], 1).drop(['age'], 1)
data3.to_csv("childAllAs.csv", sep = ",", index=False)


data4 = data2.drop(['ethnicity'], 1).drop(['country_of_res'], 1).drop(['used_app_before'], 1).\
drop(['age_desc'], 1).drop(['relation'], 1)#.drop(['age'], 1)#.drop(['result'], 1)
data4.to_csv("childDataNaBinaryNAgeWResult.csv", sep = ",", index=False)

data5 = data4[['A5', 'A6', 'A9', 'A10', 'age', 'gender', 'jundice', 'autism', 'class']]
#data5 = data4[['A5', 'A6', 'A9', 'A10', 'class']]
data5.to_csv("childDataNaAgeNResult.csv", sep = ",", index=False)