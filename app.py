import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import time

# Fungsi untuk melatih model
@st.cache_resource
def train_model(data, features, target):
    scaler = MinMaxScaler()
    X = data[features]
    y = data[target]
    X_scaled = scaler.fit_transform(X)
    
    # Reshape data for LSTM (samples, timesteps, features)
    X_scaled = np.expand_dims(X_scaled, axis=1)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Build and compile LSTM Model
    model = Sequential([
        LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=5, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    
    return model, scaler

# Fungsi untuk prediksi berdasarkan nama pemain
def predict_by_player_name(player_name, model, scaler, data, threshold=0.5):
    player_row = data[data['player_name'].str.lower() == player_name.lower()]
    if not player_row.empty:
        days = player_row['Days'].values[0]
        games_missed = player_row['Games missed'].values[0]
        
        player_data_scaled = scaler.transform([[days, games_missed]])
        player_data_scaled = np.expand_dims(player_data_scaled, axis=1)
        
        risk_score = model.predict(player_data_scaled)[0][0]
        if risk_score > threshold:
            predicted_class = "High Injury Risk"
            recommendation = "Disarankan untuk tidak memainkan pemain."
        else:
            predicted_class = "Low Injury Risk"
            recommendation = "Pemain dapat dimainkan, tetapi perhatikan kondisinya."
        
        return predicted_class, risk_score, recommendation
    else:
        return None, None, "Pemain tidak ditemukan"

# Streamlit App
st.title("Prediksi Risiko Cedera Pemain")

# Load dan proses data
file_path = r'C:\22.11.5231\AI lanjut\streamlit-injury-risk\Injuries.xlsx'
data = pd.read_excel(file_path)

# Data Cleaning
data['Days'] = data['Days'].str.replace(' days', '', regex=False).replace('?', np.nan).astype(float)
data['Games missed'] = data['Games missed'].replace(['?', '-'], np.nan).astype(float)
data['Days'].fillna(data['Days'].median(), inplace=True)
data['Games missed'].fillna(data['Games missed'].median(), inplace=True)

# Create target column 'High Injury Risk'
data['High Injury Risk'] = ((data['Days'] > 30) | (data['Games missed'] > 5)).astype(int)

# Latih model dengan caching dan tampilkan pesan dinamis
status_placeholder = st.empty()  # Placeholder untuk status
status_placeholder.text("📊 Melatih model, harap tunggu...")  # Pesan saat proses pelatihan

model, scaler = train_model(data, ['Days', 'Games missed'], 'High Injury Risk')

status_placeholder.success("✅ Model berhasil dilatih!")  # Ubah pesan setelah selesai

# Tunggu beberapa detik sebelum menghapus pesan
time.sleep(3)
status_placeholder.empty()  # Hapus pesan setelah 3 detik

# Input dari pengguna
player_name = st.text_input("Masukkan nama pemain:")

# Gunakan session state untuk melacak tombol prediksi dan hasil
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Tombol prediksi
if st.button("Prediksi"):
    if player_name:
        predicted_class, risk_score, recommendation = predict_by_player_name(player_name, model, scaler, data)
        st.session_state.prediction_result = {
            "player_name": player_name,
            "predicted_class": predicted_class,
            "risk_score": risk_score,
            "recommendation": recommendation
        }
    else:
        st.session_state.prediction_result = {
            "player_name": None,
            "predicted_class": None,
            "risk_score": None,
            "recommendation": "Masukkan nama pemain untuk memulai prediksi."
        }

# Tampilkan hasil prediksi jika ada
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    if result["player_name"]:
        st.write(f"Nama Pemain: {result['player_name']}")
        st.write(f"Risk Score: {result['risk_score']:.4f}")
        st.write(f"Predicted Class: {result['predicted_class']}")
        st.write(f"Rekomendasi: {result['recommendation']}")
    else:
        st.warning(result["recommendation"])
