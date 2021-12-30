import data_access
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

palette = "Set2"
data = data_access.Data()
df_preprocessed = data.get_preprocessed_df()
print(df_preprocessed.groupby('AVGSalesCategorical')['index'].count())
cmap = plt.get_cmap(palette)
plt.pie(df_preprocessed.groupby('AVGSalesCategorical')['index'].count(), colors=cmap(np.array([0, 1, 2, 3])))
plt.legend(["1 :  < 4413 $","2 : < 5460 $","3 : < 6634 $","4 : < 20825 $"])
plt.show()

sns.set_theme()
sns.histplot(data=df_preprocessed, x="Sales", hue="AVGSalesCategorical", palette=palette)
plt.show()