import pandas as pd

df = pd.read_parquet("mimic_albanevial/data/df_train.parquet")
print(df["admission_type"].value_counts())