from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import random
from operator import itemgetter
from datetime import datetime, timedelta

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# LSTM Model definition
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Load model and scaler
def load_model(model_path, device='cpu'):
    # Load the full checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if it contains the expected structure
    if "model_state_dict" in checkpoint:
        # If the model was saved as a full checkpoint
        state_dict = checkpoint["model_state_dict"]
        # Get hyperparameters if they exist
        hidden_size = checkpoint.get("hidden_size", 64)
        num_layers = checkpoint.get("num_layers", 2)
    else:
        # Assume it's just the state dict
        state_dict = checkpoint
        hidden_size = 64
        num_layers = 2
    
    # Create model with the right architecture
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=1)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Global scaler, load this when the application starts
scaler = None
model = None
model_path = "lstm_stock_model.pt"  # Update this path
scaler_path = "scaler.pkl"  # Update this path

# Hardcoded cryptocurrency data (last 30 days of prices)
CRYPTO_DATA = {
    "BTC": {
        "name": "Bitcoin",
        "data": [45200, 45800, 46700, 46500, 47300, 48200, 47900, 47500, 48600, 49100, 
                 48900, 49500, 50200, 49800, 49600, 51000, 52300, 52100, 51800, 53000, 
                 54200, 54500, 54000, 55200, 56500, 56100, 55800, 57200, 57500, 58000],
        "description": "The original cryptocurrency with the highest market cap"
    },
    "ETH": {
        "name": "Ethereum",
        "data": [2800, 2850, 2900, 2920, 2980, 3050, 3100, 3080, 3150, 3200,
                 3190, 3250, 3300, 3280, 3320, 3400, 3450, 3420, 3500, 3580,
                 3600, 3650, 3680, 3720, 3800, 3780, 3850, 3900, 3950, 4000],
        "description": "Smart contract platform with the second largest market cap"
    },
    "ADA": {
        "name": "Cardano",
        "data": [0.52, 0.54, 0.53, 0.55, 0.57, 0.56, 0.58, 0.59, 0.61, 0.60,
                 0.62, 0.64, 0.63, 0.65, 0.67, 0.66, 0.68, 0.70, 0.69, 0.71,
                 0.73, 0.75, 0.74, 0.76, 0.78, 0.77, 0.79, 0.81, 0.82, 0.84],
        "description": "Proof-of-stake blockchain platform with focus on sustainability"
    },
    "SOL": {
        "name": "Solana",
        "data": [95, 97, 96, 99, 103, 101, 105, 107, 106, 109,
                 112, 114, 113, 116, 120, 118, 122, 125, 123, 127,
                 130, 132, 134, 136, 135, 138, 142, 145, 148, 150],
        "description": "High-performance blockchain with low transaction fees"
    },
    "DOT": {
        "name": "Polkadot",
        "data": [5.80, 5.85, 5.90, 5.88, 5.95, 6.05, 6.10, 6.08, 6.15, 6.20,
                 6.25, 6.30, 6.28, 6.35, 6.40, 6.38, 6.45, 6.50, 6.55, 6.60,
                 6.65, 6.70, 6.75, 6.72, 6.80, 6.85, 6.90, 7.00, 7.05, 7.10],
        "description": "Multi-chain network enabling cross-blockchain transfers"
    },
    "XRP": {
        "name": "Ripple",
        "data": [0.56, 0.57, 0.58, 0.575, 0.59, 0.60, 0.595, 0.61, 0.62, 0.625,
                 0.63, 0.64, 0.635, 0.65, 0.66, 0.655, 0.67, 0.68, 0.685, 0.69,
                 0.70, 0.71, 0.705, 0.72, 0.73, 0.725, 0.74, 0.75, 0.76, 0.77],
        "description": "Digital payment protocol and cryptocurrency"
    },
    "LINK": {
        "name": "Chainlink",
        "data": [12.5, 12.6, 12.7, 12.65, 12.8, 12.9, 12.85, 13.0, 13.1, 13.05,
                 13.2, 13.3, 13.25, 13.4, 13.5, 13.45, 13.6, 13.7, 13.8, 13.75,
                 13.9, 14.0, 14.1, 14.05, 14.2, 14.3, 14.25, 14.4, 14.5, 14.6],
        "description": "Decentralized oracle network for blockchain data"
    },
    "AVAX": {
        "name": "Avalanche",
        "data": [
    9252.27734375, 9428.3330078125, 9277.9677734375, 9278.8076171875, 9240.3466796875,
    9276.5, 9243.6142578125, 9243.2138671875, 9192.8369140625, 9132.2275390625,
    9151.392578125, 9159.0400390625, 9185.8173828125, 9164.2314453125, 9374.8876953125,
    9525.36328125, 9581.072265625, 9536.892578125, 9677.11328125, 9905.1669921875,
    10990.873046875, 10912.8232421875, 11100.4677734375, 11111.2138671875, 11323.466796875,
    11759.5927734375, 11053.6142578125, 11246.3486328125, 11205.892578125, 11747.022460937
],
        "description": "Layer one blockchain focusing on speed and low cost"
    },
    "MATIC": {
        "name": "Polygon",
        "data": [0.85, 0.86, 0.87, 0.865, 0.88, 0.89, 0.885, 0.90, 0.91, 0.905,
                 0.92, 0.93, 0.925, 0.94, 0.95, 0.945, 0.96, 0.97, 0.98, 0.975,
                 0.99, 1.00, 1.01, 1.005, 1.02, 1.03, 1.025, 1.04, 1.05, 1.06],
        "description": "Layer 2 scaling solution for Ethereum"
    },
    "ATOM": {
        "name": "Cosmos",
        "data": [9.2, 9.3, 9.35, 9.28, 9.4, 9.5, 9.45, 9.55, 9.65, 9.6,
                 9.7, 9.8, 9.75, 9.9, 10.0, 9.95, 10.1, 10.2, 10.15, 10.3,
                 10.4, 10.35, 10.5, 10.6, 10.55, 10.7, 10.8, 10.9, 11.0, 11.1],
        "description": "Ecosystem of interoperable blockchains"
    }
}

def load_resources():
    global scaler, model
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print(f"Model file not found at {model_path}")

def predict_future(model, input_sequence, num_days, scaler, device='cpu'):
    """
    Predict future values using the trained model with adjustments
    for more realistic-looking predictions with a slight upward trend
    """
    model.eval()
    future_predictions = []
    daily_returns = []
    
    # Make sure input is a tensor and has the right shape
    current_sequence = torch.FloatTensor(input_sequence).to(device)
    
    # Calculate the average change of input sequence
    avg_change = torch.mean(torch.abs(current_sequence[1:] - current_sequence[:-1])).item()
    
    # Get the last value of the input sequence
    prev_value = current_sequence[-1].item()
    
    with torch.no_grad():
        for i in range(num_days):
            # Reshape for model input [batch_size, sequence_length, features]
            x = current_sequence.view(1, -1, 1)
            
            # Get raw prediction
            raw_pred = model(x).item()
            
            # Add a slight upward trend (0.1-0.3% daily increase on average)
            upward_bias = prev_value * (5e-7 + 0.002 * np.random.random())
            
            # Calculate volatility factor based on day number with some randomness
            volatility = avg_change * (0.8 + 0.5 * np.sin(i/5) + 0.3 * np.random.random())
            
            # Adjust prediction to create a more natural-looking pattern with upward bias
            if i < num_days / 3:
                noise = volatility * np.sin(i/2) * 0.5
                adjusted_pred = raw_pred + noise + upward_bias
            elif i < 2 * num_days / 3:
                mean_reversion = (np.mean(input_sequence) - raw_pred) * (i - num_days/3) / (num_days/3) * 0.2
                noise = volatility * np.sin(i/2) * 0.7
                adjusted_pred = raw_pred + mean_reversion + noise + upward_bias * 1.2
            else:
                trend_factor = upward_bias * 1.5
                noise = volatility * np.sin(i/1.5) * 1.1
                adjusted_pred = raw_pred + noise + trend_factor
            
            # Ensure we don't get negative values for prices
            adjusted_pred = max(adjusted_pred, 0.01)
            
            # Calculate daily return for volatility measurement
            if i > 0:
                daily_return = (adjusted_pred - prev_value) / prev_value
                daily_returns.append(daily_return)
            
            # Update previous value for next iteration
            prev_value = adjusted_pred
            
            # Append to our predictions
            future_predictions.append(adjusted_pred)
            
            # Update the sequence
            new_pred_tensor = torch.tensor([adjusted_pred], dtype=torch.float32, device=device)
            current_sequence = torch.cat((current_sequence[1:], new_pred_tensor))
    
    # Convert predictions to numpy array and reshape for inverse transform
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    
    # Inverse transform to get the actual values
    future_predictions_rescaled = scaler.inverse_transform(future_predictions).flatten()
    
    # Calculate volatility assessment
    if daily_returns:
        volatility_value = np.std(daily_returns) * np.sqrt(252)  # Annualized volatility
        volatility_assessment = "High" if volatility_value > 0.25 else "Medium" if volatility_value > 0.15 else "Low"
    else:
        volatility_assessment = "Unknown"
    
    return future_predictions_rescaled.tolist(), volatility_assessment

def forecast_n_days(model, past_30_days, n_days, custom_scaler=None, device='cpu'):
    """
    Take the past 30 days as input and predict the next n days
    """
    # Use global scaler if none provided
    if custom_scaler is None:
        custom_scaler = scaler
    
    # Ensure input is numpy array with the right shape
    if isinstance(past_30_days, list):
        past_30_days = np.array(past_30_days)
    
    if len(past_30_days) != 30:
        raise ValueError("Input data must contain exactly 30 days of data")
    
    # Reshape and scale the input data
    past_30_days_reshaped = past_30_days.reshape(-1, 1)
    past_30_days_scaled = custom_scaler.transform(past_30_days_reshaped).flatten()
    
    # Predict the next n days
    future_preds, volatility_assessment = predict_future(model, past_30_days_scaled, n_days, custom_scaler, device)
    
    # Calculate profit percentage
    initial_price = past_30_days[-1]
    max_price = np.max(future_preds)
    profit_percentage = ((max_price / initial_price) - 1) * 100
    
    return {
        "past_data": past_30_days.tolist(),
        "predictions": future_preds,
        "profit_percentage": round(profit_percentage, 2),
        "volatility_assessment": volatility_assessment,
        "initial_price": float(initial_price),
        "max_price": float(max_price)
    }

# Root endpoint to check if API is working
@app.route('/', methods=['GET', 'OPTIONS'])
def home():
    return jsonify({"message": "Crypto Prediction API is running"})

# Endpoint to get list of available cryptocurrencies
@app.route('/get_available_coins', methods=['GET'])
def get_available_coins():
    """
    Return the list of available cryptocurrency data
    """
    coins = []
    for symbol, data in CRYPTO_DATA.items():
        coins.append({
            "symbol": symbol,
            "name": data["name"],
            "description": data["description"],
            "current_price": data["data"][-1]
        })
    
    return jsonify(coins)

# New endpoint to get top performing coins
@app.route('/get_top_performers', methods=['GET'])
def get_top_performers():
    """
    Analyze all cryptocurrencies and return the top performers based on prediction
    """
    # Check if model and scaler are loaded
    if model is None or scaler is None:
        load_resources()
        if model is None or scaler is None:
            return jsonify({'error': 'Model or scaler not available'}), 500
    
    # Get the number of days to predict (default: 30)
    n_days = request.args.get('days', default=30, type=int)
    
    # Limit to reasonable range
    if n_days < 1:
        n_days = 1
    elif n_days > 90:
        n_days = 90
    
    # Get the number of top performers to return (default: 3)
    top_n = request.args.get('top_n', default=3, type=int)
    
    # Make predictions for each coin
    results = []
    
    for symbol, coin_data in CRYPTO_DATA.items():
        try:
            prediction_result = forecast_n_days(model, coin_data["data"], n_days)
            
            # Add coin information
            prediction_result["symbol"] = symbol
            prediction_result["name"] = coin_data["name"]
            prediction_result["description"] = coin_data["description"]
            
            results.append(prediction_result)
        except Exception as e:
            print(f"Error predicting {symbol}: {str(e)}")
    
    # Sort by profit percentage (descending)
    sorted_results = sorted(results, key=itemgetter('profit_percentage'), reverse=True)
    
    # Return only the top N performers
    top_performers = sorted_results[:top_n]
    
    # Generate dates for the past 30 days
    today = datetime.now()
    dates = [(today - timedelta(days=29-i)).strftime('%Y-%m-%d') for i in range(30)]
    
    # Add dates to the response
    response = {
        "prediction_days": n_days,
        "total_coins_analyzed": len(results),
        "top_performers": top_performers,
        "dates": dates
    }
    
    return jsonify(response)

# Original predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    
    if not data or 'past_30_days' not in data or 'n_days' not in data:
        return jsonify({'error': 'Missing required data'}), 400
    
    try:
        past_30_days = data['past_30_days']
        n_days = int(data['n_days'])
        
        # Check if we have a model and scaler loaded
        if model is None or scaler is None:
            load_resources()
            if model is None or scaler is None:
                return jsonify({'error': 'Model or scaler not available'}), 500
        
        # Make prediction
        result = forecast_n_days(model, past_30_days, n_days)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model and scaler when starting the application
    load_resources()
    app.run(host='0.0.0.0', port=5000, debug=True)