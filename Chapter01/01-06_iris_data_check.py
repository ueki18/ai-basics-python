from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = [iris.target_names[i] for i in iris.target]

df.info()
print(df.isnull().sum())
