# Create pandas DataFrame
import pandas as pd
import numpy as np
technologies = {
    'Courses':["Spark","PySpark","Python"],
    'Fee' :[20000,25000,22000],
    'Duration':['30days','40days','35days'],
    'Discount':[1000,2300,1200]
              }
df = pd.DataFrame(technologies)
print("Create DataFrame:\n",df.columns)

# Get column name by index
index_to_get = 2
df2 = df.columns[index_to_get]
print(f"Column name at index {index_to_get}: {df2}")

# Get column name by index
# Using the columns attribute
df2 = df.columns[[1,3]]
print(f"Column name at index {3}: {df2}")

# Get multiple column names by index
indices_to_get = [1, 3]  # List of indices
df2 = df.columns[indices_to_get]
print(f"Column names at indices {indices_to_get}: {df2}")
