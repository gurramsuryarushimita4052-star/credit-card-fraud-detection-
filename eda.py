import pandas as pd

df = pd.read_csv("dataset/creditcard.csv")

print("DATASET LOADED SUCCESSFULLY!")
print("\nFIRST FIVE ROWS")
print(df.head())

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum().sum())

print("\nCLASS DISTRIBUTION")
print(df["Class"].value_counts())

print("\nEDA COMPLETED SUCCESSFULLY!")