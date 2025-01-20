import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=1, batch_size=16, validation_data=(X_test, y_test), verbose=1)
    
    return model, scaler, X_test, y_test

# Fungsi untuk prediksi berdasarkan nama pemain
def predict_by_player_name(player_name, model, scaler, data, threshold=0.5):
    player_row = data[data['player_name'].str.lower() == player_name.lower()]
    if not player_row.empty:
        days = player_row['Days'].values[0]
        games_missed = player_row['Games missed'].values[0]

        # Pastikan data memiliki nama kolom saat menggunakan MinMaxScaler
        player_data = pd.DataFrame([[days, games_missed]], columns=['Days', 'Games missed'])
        player_data_scaled = scaler.transform(player_data)
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
file_path = 'Injuries.xlsx'
data = pd.read_excel(file_path, engine='openpyxl')

# Data Cleaning
data['Days'] = data['Days'].str.replace(' days', '', regex=False).replace(['?', ''], np.nan)
data['Days'] = pd.to_numeric(data['Days'], errors='coerce')
data['Games missed'] = data['Games missed'].replace(['?', '-', ''], np.nan)
data['Games missed'] = pd.to_numeric(data['Games missed'], errors='coerce')
data['Days'] = data['Days'].fillna(data['Days'].median())
data['Games missed'] = data['Games missed'].fillna(data['Games missed'].median())

# Create target column 'High Injury Risk'
data['High Injury Risk'] = ((data['Days'] > 30) | (data['Games missed'] > 5)).astype(int)

# Latih model dengan caching dan tampilkan pesan dinamis
status_placeholder = st.empty()  # Placeholder untuk status
status_placeholder.text(" Melatih model, harap tunggu...")  # Pesan saat proses pelatihan

model, scaler, X_test, y_test = train_model(data, ['Days', 'Games missed'], 'High Injury Risk')

# Hitung metrik evaluasi
y_pred = model.predict(X_test)
y_pred_class = (y_pred > 0.5).astype(int)

# Menghitung metrik evaluasi
accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class)
f1 = f1_score(y_test, y_pred_class)

status_placeholder .success(" Model berhasil dilatih!")  # Ubah pesan setelah selesai

# Tunggu beberapa detik sebelum menghapus pesan
time.sleep(3)
status_placeholder.empty()  # Hapus pesan setelah 3 detik

# Tampilkan metrik evaluasi
st.write("Metrik Evaluasi Model:")
st.write(f"Akurasi: {accuracy:.2f}")
st.write(f"Precision: {precision:.2f}")
st.write(f"Recall: {recall:.2f}")
st.write(f"F1 Score: {f1:.2f}")

# Input untuk prediksi berdasarkan nama pemain
player_name = st.text_input("Masukkan nama pemain:")
if st.button("Prediksi Risiko Cedera"):
    predicted_class, risk_score, recommendation = predict_by_player_name(player_name, model, scaler, data)
    if predicted_class:
        st.write(f"Prediksi Kelas: {predicted_class}")
        st.write(f"Skor Risiko: {risk_score:.2f}")
        st.write(f"Rekomendasi: {recommendation}")
    else:
        st.write(recommendation)