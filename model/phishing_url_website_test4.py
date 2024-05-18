import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
data = pd.read_csv('phishing_url_website.csv')

object_cols = [i for i in data.columns.tolist() if data[i].dtype=="object"]
print(data[object_cols])
data.drop(object_cols, axis=1, inplace=True)
top_corr = data.corrwith(data["label"]).abs().sort_values(ascending=False)[:6].index.to_list()
[data[i].value_counts() for i in data[top_corr]]

fig = plt.figure(figsize=(12, 3), dpi=200)
ax = fig.add_axes(121)
ax2 = fig.add_axes(122)


sns.countplot(data[top_corr], x="label", hue="label", ax=ax)
ax.set_title("label")

sns.histplot(data[top_corr],bins=5, x="URLSimilarityIndex", hue="label", ax=ax2)
ax2.set_title("URLSimilarityIndex")