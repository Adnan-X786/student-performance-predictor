import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset
data = pd.read_csv("data.csv")

X = data[['study_hours', 'attendance', 'assignments']]
y = data['score']

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained successfully!")
