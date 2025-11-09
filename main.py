import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the dataset
data = pd.read_csv('new.csv')

# Extract the required features
df = data[['age', 'height', 'weight', 'size']]

# Process the features
x = df.iloc[:, :3]
y = df.iloc[:, 3]

# Initialize the Random Forest model
random_forest = RandomForestRegressor()

# Train the Random Forest model
random_forest.fit(x, y)

# Save the model in the current directory using pickle
pickle.dump(random_forest, open('hmodel.pkl', 'wb'))

