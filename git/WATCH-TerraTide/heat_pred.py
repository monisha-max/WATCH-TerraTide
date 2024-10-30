from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import random
import os

app = Flask(__name__)

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Load and preprocess data when the app starts
def load_and_preprocess_data():
    # Load CSV files
    df1 = pd.read_csv('Bronx_1.csv', encoding='latin1')
    df2 = pd.read_csv('Bronx_2.csv', encoding='latin1')
    df3 = pd.read_csv('Bronx_3.csv', encoding='latin1')
    df_combined = pd.concat([df1, df2, df3], ignore_index=True)
    
    required_features = [
        'Date & Time',
        'High Temp - °F',
        'High Hum - %',
        'High Heat Index - °F',
        'High Dew Point - °F',
        'High THW Index - °F',
        'High THSW Index - °F',
        'High Solar Rad - W/m^2',
        'High UV Index',
        'High Wind Speed - mph',
        'Cooling Degree Days'
    ]
    df_combined = df_combined[required_features]
    df_combined['Date & Time'] = pd.to_datetime(df_combined['Date & Time'], errors='coerce')
    df_combined.set_index('Date & Time', inplace=True)
    
    for col in required_features[1:]:
        df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')
        df_combined[col].fillna(df_combined[col].mean(), inplace=True)
    
    df_resampled = df_combined.resample('6H').mean()
    df_resampled.reset_index(inplace=True)
    df_resampled['Month'] = df_resampled['Date & Time'].dt.month
    df_resampled['Year'] = df_resampled['Date & Time'].dt.year

    # Fill any remaining NaN values
    columns_to_fill = [
        'High Temp - °F',
        'High Hum - %',
        'High Heat Index - °F',
        'High Dew Point - °F',
        'High THW Index - °F',
        'High THSW Index - °F',
        'High Solar Rad - W/m^2',
        'High UV Index',
        'High Wind Speed - mph',
        'Cooling Degree Days'
    ]
    for col in columns_to_fill:
        mean_value = df_resampled[col].mean()
        df_resampled[col].fillna(mean_value, inplace=True)

    return df_resampled

# Function to train the model
def train_model(data):
    data.sort_values('Date & Time', inplace=True)
    data.reset_index(drop=True, inplace=True)
    data_model = data[['Date & Time', 'High Temp - °F']].copy()
    data_model.set_index('Date & Time', inplace=True)
    data_model['Month'] = data_model.index.month
    data_model['DayOfYear'] = data_model.index.dayofyear
    data_model['sin_day'] = np.sin(2 * np.pi * data_model['DayOfYear'] / 365)
    data_model['cos_day'] = np.cos(2 * np.pi * data_model['DayOfYear'] / 365)

    # Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_model[['High Temp - °F', 'Month', 'sin_day', 'cos_day']])

    # Create sequences
    n_past = 300
    n_future = 60
    X, y = [], []
    for i in range(n_past, len(scaled_data) - n_future + 1):
        X.append(scaled_data[i - n_past:i])
        y.append(scaled_data[i + n_future - 1, 0])  # 'High Temp - °F' is the first feature
    X, y = np.array(X), np.array(y)

    # Split data
    validation_split = 0.1
    split_index = int(len(X) * (1 - validation_split))
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    # Build model
    model = Sequential()
    model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.3))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0  # Set verbose to 0 to suppress training output
    )

    # Save the scaler and model
    model.save('lstm_model.h5')
    np.save('scaler.npy', scaler.scale_)
    np.save('scaler_min.npy', scaler.min_)
    return model, scaler, data_model

# Function to make future predictions
def make_predictions(model, scaler, data_model):
    n_past = 300
    n_future = 60

    scaled_data = scaler.transform(data_model[['High Temp - °F', 'Month', 'sin_day', 'cos_day']])
    last_sequence = scaled_data[-n_past:].copy()

    future_predictions = []
    current_sequence = last_sequence.copy()

    for i in range(n_future):
        current_sequence_reshaped = current_sequence.reshape(1, n_past, current_sequence.shape[1])
        next_pred_scaled = model.predict(current_sequence_reshaped)[0, 0]
        future_predictions.append(next_pred_scaled)

        next_day = pd.to_datetime(data_model.index[-1]) + pd.Timedelta(days=i + 1)
        sin_day = np.sin(2 * np.pi * next_day.timetuple().tm_yday / 365)
        cos_day = np.cos(2 * np.pi * next_day.timetuple().tm_yday / 365)
        next_feature = np.array([next_pred_scaled, next_day.month, sin_day, cos_day])

        current_sequence = np.vstack((current_sequence[1:], next_feature))

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions_unscaled = scaler.inverse_transform(np.hstack((future_predictions, np.zeros((len(future_predictions), 3)))))[:, 0]

    future_dates = pd.date_range(start=data_model.index[-1] + pd.Timedelta(days=1), periods=n_future)
    future_dates = future_dates.date

    forecast_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted High Temp - °F': future_predictions_unscaled
    })

    return forecast_df

# Function to generate heatwave warnings
def generate_heatwave_warnings(forecast_df, data_model):
    heat_wave_threshold = 90

    heatwave_warnings = {
        "Within 2 days": False,
        "Within 5 days": False,
        "Within 2 weeks": False,
        "Within 1 month": False,
        "Within 2 months": False
    }

    precautions = {
        "Within 2 days": "There are chances of happening heat waves within 2 days. Drink plenty of water and use fans or air conditioning to lower body temperature. Avoid strenuous outdoor activities during peak heat hours (10 AM - 4 PM).",
        "Within 5 days": "There are chances of happening heat waves within 5 days. Ensure elderly neighbors, children, and pets are safe and have access to cooling. Reschedule or cancel non-essential outdoor activities during extreme heat.",
        "Within 2 weeks": "There are chances of happening heat waves within 2 weeks. Coordinate with local authorities to open cooling centers and inform the public about heat safety tips.",
        "Within 1 month": "There are chances of happening heat waves within 1 month. Consider rescheduling outdoor events, ensure cooling measures at home.",
        "Within 2 months": "There are chances of happening heat waves within 2 months. Prepare for prolonged heat with adequate supplies and cooling strategies."
    }

    last_recorded_date = data_model.index[-1]
    dates_within_2_days = last_recorded_date + pd.Timedelta(days=2)
    dates_within_5_days = last_recorded_date + pd.Timedelta(days=5)
    dates_within_2_weeks = last_recorded_date + pd.Timedelta(weeks=2)
    dates_within_1_month = last_recorded_date + pd.Timedelta(weeks=4)
    dates_within_2_months = last_recorded_date + pd.Timedelta(weeks=8)

    for i, row in forecast_df.iterrows():
        if row['Predicted High Temp - °F'] > heat_wave_threshold:
            forecast_date = pd.to_datetime(row['Date'])
            if forecast_date <= dates_within_2_days:
                heatwave_warnings["Within 2 days"] = True
            if forecast_date <= dates_within_5_days:
                heatwave_warnings["Within 5 days"] = True
            if forecast_date <= dates_within_2_weeks:
                heatwave_warnings["Within 2 weeks"] = True
            if forecast_date <= dates_within_1_month:
                heatwave_warnings["Within 1 month"] = True
            if forecast_date <= dates_within_2_months:
                heatwave_warnings["Within 2 months"] = True

    warnings = []
    for period, is_heatwave in heatwave_warnings.items():
        if is_heatwave:
            precaution_message = precautions.get(period, "No specific precaution available.")
            heatwave_dates = forecast_df[(forecast_df['Predicted High Temp - °F'] > heat_wave_threshold) & 
                                         (pd.to_datetime(forecast_df['Date']) <= eval(f"dates_within_{period.replace(' ', '_')}"))]['Date'].tolist()
            warnings.append({
                'period': period,
                'dates': heatwave_dates,
                'precaution': precaution_message
            })

    if not warnings:
        warnings.append({
            'period': 'No Heatwave',
            'dates': [],
            'precaution': 'There is no heatwave coming in the next 2 months.'
        })

    return warnings

# Route for home page
@app.route('/')
def home():
    return render_template('index3.html')

# Route to trigger prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess data
    data = load_and_preprocess_data()

    # Check if model exists
    if os.path.exists('lstm_model.h5') and os.path.exists('scaler.npy') and os.path.exists('scaler_min.npy'):
        from tensorflow.keras.models import load_model
        model = load_model('lstm_model.h5')
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.scale_ = np.load('scaler.npy')
        scaler.min_ = np.load('scaler_min.npy')
        scaler.data_min_ = np.zeros(4)
        scaler.data_max_ = np.ones(4)
        scaler.data_range_ = scaler.data_max_ - scaler.data_min_
        # Prepare data_model
        data.sort_values('Date & Time', inplace=True)
        data.reset_index(drop=True, inplace=True)
        data_model = data[['Date & Time', 'High Temp - °F']].copy()
        data_model.set_index('Date & Time', inplace=True)
        data_model['Month'] = data_model.index.month
        data_model['DayOfYear'] = data_model.index.dayofyear
        data_model['sin_day'] = np.sin(2 * np.pi * data_model['DayOfYear'] / 365)
        data_model['cos_day'] = np.cos(2 * np.pi * data_model['DayOfYear'] / 365)
    else:
        # Train model
        model, scaler, data_model = train_model(data)

    # Make predictions
    forecast_df = make_predictions(model, scaler, data_model)
    warnings = generate_heatwave_warnings(forecast_df, data_model)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(data_model.index, data_model['High Temp - °F'], label='Historical Temperature')
    plt.plot(forecast_df['Date'], forecast_df['Predicted High Temp - °F'], label='Forecasted Temperature (LSTM)', color='red')
    plt.title('Temperature Forecast with LSTM')
    plt.xlabel('Date')
    plt.ylabel('High Temp - °F')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('index3.html', 
                           plot_url=plot_url,
                           forecast_table=forecast_df.to_html(classes='table table-striped', index=False),
                           warnings=warnings)

if __name__ == '__main__':
    app.run(debug=True,port=5002)
