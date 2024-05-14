# Project which analyses random social data and conclusion of it.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random

#creating random data

categories = ["Food", "Travel", "Fashion", "Music", "Culture", "Family", "Health"]
data = {"Date": pd.date_range("2024-01-01", periods=500),
        "Category": [random.choice(categories) for n in range(500)],
        "Likes": np.random.randint(0, 10000, size = 500)
        }

df = pd.DataFrame(data)

#show 20 values at the head
print(df.head(10))
#show brief information abput dataframe
print(df.info())
#description for dataframe
print(df.describe())
#value counting by column Category
print(df.value_counts(subset= "Category"))

#cleaning the data
#droping null values if they exist
df.dropna()

#droping duplicate values if they exist
df.drop_duplicates()

#converting date format to datetime
pd.to_datetime(df["Date"])

#converting type of 'likes' column to integer
df["Likes"].astype(dtype=int)

#visualising data in histogram and boxplot
sns.histplot(data["Likes"])
plt.show()

sns.boxplot(data=df, x="Category", y="Likes")
plt.show()

#finding average of all of 'Likes' and the average per category in 'Category' 
mean = df["Likes"].mean()
print(f"Mean of the 'Likes' column: {mean}")

print(df.groupby('Category')["Likes"].mean())










