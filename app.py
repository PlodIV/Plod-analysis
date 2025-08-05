import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

st.title("📈 ASX 200 Daily Forecasting Tool")

@st.cache_data
def load_data():
    asx = yf.download("^AXJO", start="2015-01-01")
    sp500 = yf.download("^GSPC", start="2015-01-01")
    audusd = yf.download("AUDUSD=X", start="2015-01-01")
    oil = yf.download("CL=F", start="2015-01-01")
    
    df = asx.copy()
    df['Return'] = df['Close'].pct_change()
    df['SP500'] = sp500['Close'].pct_change()
    df['AUDUSD'] = audusd['Close'].pct_change()
    df['OIL'] = oil['Close'].pct_change()
    
    df['RSI'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['MACD'] = MACD(close=df['Close']).macd_diff()
    df['Lag1'] = df['Return'].shift(1)
    df['Lag2'] = df['Return'].shift(2)
    df['Direction'] = (df['Return'].shift(-1) > 0).astype(int)
    
    return df.dropna()

df = load_data()
st.subheader("Market Overview")
st.line_chart(df['Close'])

# Feature selection
features = ['Lag1', 'Lag2', 'RSI', 'MACD', 'SP500', 'AUDUSD', 'OIL']
X = df[features]
y = df['Direction']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Model selection
model_name = st.sidebar.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"])
if model_name == "Logistic Regression":
    model = LogisticRegression()
elif model_name == "Random Forest":
    model = RandomForestClassifier()
else:
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.write(f"**Accuracy:** {acc:.2f}")
st.text(classification_report(y_test, y_pred))

latest_input = X.iloc[-1:]
prediction = model.predict(latest_input)[0]
st.subheader("📊 Next-Day Forecast")
st.write(f"**Prediction:** {'📈 UP' if prediction == 1 else '📉 DOWN'}")

# Plot RSI and MACD
st.subheader("Technical Indicators")
fig, ax = plt.subplots()
df[['RSI', 'MACD']].tail(100).plot(ax=ax)
st.pyplot(fig)