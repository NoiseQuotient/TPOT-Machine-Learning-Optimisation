import yfinance as yf
import pandas as pd

# Download data for a stock (e.g., Apple) for the last 1 year
symbol = 'AAPL'
data = yf.download(symbol, period="1y", interval="1d")

# Check the first few rows of the data
print(data.head())



# Calculate daily returns
data['Returns'] = data['Close'].pct_change()

# Calculate moving averages
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# Drop missing values
data.dropna(inplace=True)

# Display the data
print(data.tail())


# Create the target variable: 1 if price goes up tomorrow, 0 if it goes down
data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Select the features
X = data[['Returns', 'MA20', 'MA50']]
y = data['Target']


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')



import tpot
print(tpot.__version__)




import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

# Download stock data (replace 'AAPL' with the symbol of your choice)
df = yf.download('AAPL', start='2010-01-01', end='2023-01-01')

# Create the 'Target' column based on future price movement
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 if price goes up, 0 if price goes down

# Choose your features (e.g., 'Open', 'High', 'Low', 'Close', 'Volume')
features = ['Open', 'High', 'Low', 'Close', 'Volume']
X = df[features]
y = df['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize TPOT for AutoML
tpot = TPOTClassifier(generations=3, population_size=10, random_state=42, max_time_mins=2)
  # Max time of 60 minutes

# Train the model using TPOT
tpot.fit(X_train, y_train)

# Evaluate on the test set
print(f"Test Score: {tpot.score(X_test, y_test):.4f}")

# Export the best model found by TPOT
tpot.export('best_model_pipeline.py')

