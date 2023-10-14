## Ex06-Feature Transformation
## AIM
To read the given data and perform Feature Transformation process and save the data to a file.

## EXPLANATION
Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

## ALGORITHM
STEP 1: Read the given Data

STEP 2: Clean the Data Set using Data Cleaning Process

STEP 3: Apply Feature Transformation techniques to all the features of the data set

STEP 4: Print the transformed features

## PROGRAM
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.preprocessing import QuantileTransformer

df=pd.read_csv("data_trans.csv")
df

sm.qqplot(df['HighlyPositiveSkew'],fit=True,line='45')
plt.show()

sm.qqplot(df['HighlyNegativeSkew'],fit=True,line='45')
plt.show()

sm.qqplot(df['ModeratePositiveSkew'],fit=True,line='45')
plt.show()

sm.qqplot(df['ModerateNegativeSkew'],fit=True,line='45')
plt.show()

df['HighlyPositiveSkew']=np.log(df.HighlyPositiveSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['HighlyNegativeSkew']=np.log(df.HighlyNegativeSkew)
sm.qqplot(df.HighlyPositiveSkew,fit=True,line='45')
plt.show()

df['ModeratePositiveSkew_1'], parameters=stats.yeojohnson(df.ModeratePositiveSkew)
sm.qqplot(df.ModeratePositiveSkew_1,fit=True,line='45')
plt.show()

df['ModerateNegativeSkew_1'], parameters=stats.yeojohnson(df.ModerateNegativeSkew)
sm.qqplot(df.ModerateNegativeSkew_1,fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['ModerateNegativeSkew']]))
sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt= QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2']=pd.DataFrame(qt.fit_transform(df[['ModerateNegativeSkew']]))

sm.qqplot(df.ModerateNegativeSkew_2,fit=True,line='45')
plt.show()

df2=df.copy()

df2['HighlyPositiveSkew']= 1/df2.HighlyPositiveSkew
sm.qqplot(df2.HighlyPositiveSkew,fit=True,line='45')

plt.show()
```
## OUTPUT

![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/b6bc9425-59cb-447e-9084-3e41f0d21dea)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/83e6b1b3-c1bb-499a-bc56-a83064fe8170)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/b0f85378-f826-4160-b9ab-3161a3e51646)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/d35492ce-f150-45d8-a0f5-2e014d02ee5f)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/1bedd483-7562-417a-8fd9-c7b9d6184050)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/019abe57-932d-4d1e-8376-d9a769d8560f)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/37c250ff-1bfd-4508-a7a4-2f344944fb8b)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/2a5a9bd5-5f54-4876-b43a-651567570e61)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/cc2729c2-ca8e-4cb0-9417-dd51f25385d9)
![image](https://github.com/pradeepasri26/ODD2023-Datascience-Ex06/assets/131433142/e497348b-55b1-4245-94e9-77b2cca60606)

## RESULT
Thus feature transformation is done for the given dataset.
