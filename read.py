import pandas as pd

# Read the dataset
movies = pd.read_csv('dataset.csv')

# Check the columns
print(movies.columns)

# Display the first few rows
print(movies.head())
