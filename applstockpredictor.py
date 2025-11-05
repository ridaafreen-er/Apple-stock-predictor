# applstockpredictor.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

class StockPricePredictor:
    @staticmethod
    def load_data(csv_path='AAPL_data.csv'):
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        data = pd.read_csv(csv_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.sort_values('Date').reset_index(drop=True)
        return data

    @staticmethod
    def preprocess_data(data):
        # use a separate df variable
        df = data.copy()
        df['Prev_Close'] = df['Close'].shift(1)
        df = df.dropna().reset_index(drop=True)
        X = df[['Prev_Close']]
        y = df['Close']
        # preserve dates for plotting later
        dates = df['Date']
        return X, y, dates

    @staticmethod
    def split_data(X, y, dates=None, train_size=0.8):
        split_index = int(len(X) * train_size)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        if dates is not None:
            dates_train, dates_test = dates[:split_index], dates[split_index:]
            return X_train, X_test, y_train, y_test, dates_train, dates_test
        return X_train, X_test, y_train, y_test, None, None

    @staticmethod
    def train_model(X_train, y_train):
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model

    @staticmethod
    def evaluate_model(model, X_test, y_test):
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse, predictions

    @staticmethod
    def plot_results(dates_test, y_test, predictions):
        plt.figure(figsize=(10,5))
        # if dates provided, plot against dates, else use index
        if dates_test is not None:
            plt.plot(dates_test, y_test.values, label='Actual Prices')
            plt.plot(dates_test, predictions, label='Predicted Prices')
            plt.gcf().autofmt_xdate()
        else:
            plt.plot(y_test.index, y_test.values, label='Actual Prices')
            plt.plot(y_test.index, predictions, label='Predicted Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Apple Stock Price Prediction')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    try:
        print("Loading data...")
        data = StockPricePredictor.load_data('AAPL_data.csv')

        print("Preprocessing data...")
        X, y, dates = StockPricePredictor.preprocess_data(data)

        print("Splitting data...")
        X_train, X_test, y_train, y_test, dates_train, dates_test = StockPricePredictor.split_data(X, y, dates, train_size=0.8)

        print("Training model...")
        model = StockPricePredictor.train_model(X_train, y_train)

        print("Evaluating model...")
        mse, predictions = StockPricePredictor.evaluate_model(model, X_test, y_test)
        print(f"Mean Squared Error: {mse:.6f}")

        # Save predictions to excel
        out_df = pd.DataFrame({
            'Date': dates_test.reset_index(drop=True),
            'Actual': y_test.reset_index(drop=True),
            'Predicted': predictions
        })
        out_excel = 'predictions.xlsx'
        out_df.to_excel(out_excel, index=False)
        print(f"Predictions saved to {out_excel}")

        print("Plotting results...")
        StockPricePredictor.plot_results(dates_test, y_test, predictions)

    except Exception as e:
        print("ERROR:", e)
        raise
