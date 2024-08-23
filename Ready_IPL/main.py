import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("train.csv/train.csv")
pd.set_option('display.max_columns', None)
# print(df.head())
# print(df.sample(10))
df["is_duplicate"].value_counts().plot(kind="bar")
plt.show()
ques=pd.Series(df["qid1"].tolist()+df["qid2"].tolist())
print(np.unique(ques).shape[0])