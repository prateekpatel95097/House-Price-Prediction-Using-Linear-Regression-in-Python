import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# Step 1: Create Dataset
data = {
    'Size': [1500, 1800, 2400, 3000, 3500],
    'Bedrooms': [3, 4, 3, 5, 4],
    'Age': [10, 5, 3, 2, 1],
    'Price': [250000, 320000, 450000, 540000, 580000]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 2: Split Dataset
X = df[['Size', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Predict the Price of a New House
new_house = [[2000, 3, 5]]
predicted_price = model.predict(new_house)

# Step 5: Output
print("\nPredicted Price:", predicted_price[0])
