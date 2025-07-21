import streamlit as st
import sqlite3
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DB_NAME = "usd_cop_data.db"

# --- 1) Ensure our new performance_history table exists (fresh each run) ---
def ensure_performance_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS performance_history (
            prediction_date     TEXT PRIMARY KEY,
            predicted_bucket    TEXT,
            actual_bucket       TEXT,
            success             INTEGER,
            note                TEXT
        )
    """)
    conn.commit()
    conn.close()
ensure_performance_table()

# --- 2) New function: append a fresh performance row each time we fetch ---
def append_performance_row(pred_date, predicted_bucket):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # 1) clear out any prior rows
    c.execute("DELETE FROM performance_history")

    # 2) re-compute actual & success off the freshest exchange_rates
    c.execute("SELECT exchange_rate FROM exchange_rates WHERE start_date = ?", (pred_date,))
    row = c.fetchone()
    if row:
        decimal = row[0] % 1
        if decimal < 0.34:
            actual = 'bajo'
        elif decimal < 0.67:
            actual = 'medio'
        else:
            actual = 'alto'
        success = int(actual == predicted_bucket)
    else:
        actual = None
        success = None

    # 3) insert only this one record
    c.execute("""
        INSERT OR REPLACE INTO performance_history
          (prediction_date, predicted_bucket, actual_bucket, success)
        VALUES (?, ?, ?, ?)
    """, (pred_date, predicted_bucket, actual, success))

    conn.commit()
    conn.close()


# --- Function: Fetch and insert today's USD-COP rate ---
def fetch_and_store_today_rate(db_name=DB_NAME):
    url = ("https://www.datos.gov.co/resource/32sa-8pi3.json?"
           "$query=SELECT%20%60valor%60,%20%60vigenciadesde%60,%20%60vigenciahasta%60"
           "%20ORDER%20BY%20%60vigenciadesde%60%20DESC%20NULL%20LAST%20LIMIT%201")

    response = requests.get(url)
    data = response.json()

    if not data:
        return None, "No data returned from Banrep API."

    record = data[0]
    date_str = record['vigenciadesde'][:10]
    rate = float(record['valor'])

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Ensure table exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS exchange_rates (
            start_date TEXT PRIMARY KEY,
            end_date TEXT,
            exchange_rate REAL
        )
    """)

    # Check for duplicates
    cursor.execute("SELECT 1 FROM exchange_rates WHERE start_date = ?", (date_str,))
    exists = cursor.fetchone()

    if not exists:
        cursor.execute(
            "INSERT INTO exchange_rates (start_date, end_date, exchange_rate) VALUES (?, ?, ?)",
            (date_str, record['vigenciahasta'][:10], rate)
        )
        conn.commit()
        conn.close()
        return rate, "✅ Today's rate added to the database."
    else:
        conn.close()
        return rate, "ℹ️ Today's rate is already in the database."


# --- Function: Load recent DB records ---
def load_latest_rates(n=5, db_name=DB_NAME):
    conn = sqlite3.connect(db_name)
    df = pd.read_sql_query(f"""
        SELECT * FROM exchange_rates
        ORDER BY start_date DESC
        LIMIT {n}
    """, conn)
    conn.close()
    return df


# --- Function: Train model and prepare tomorrow's prediction ---
def train_model(db_path=DB_NAME):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY start_date", conn)
    conn.close()

    df['decimal'] = df['exchange_rate'] % 1

    def bucketize(decimal):
        if decimal < 0.34:
            return 'bajo'
        elif decimal < 0.67:
            return 'medio'
        else:
            return 'alto'

    df['bucket'] = df['decimal'].apply(bucketize)
    df['bucket_label'] = df['bucket'].map({'bajo': 0, 'medio': 1, 'alto': 2})

    df = df.sort_values("start_date").reset_index(drop=True)
    df['lag_1'] = df['exchange_rate'].shift(1)
    df['lag_2'] = df['exchange_rate'].shift(2)
    df['pct_change_1d'] = df['lag_1'].pct_change()
    df['day_of_week'] = pd.to_datetime(df['start_date']).dt.dayofweek
    
    # ⚠️ Shift labels 2 days ahead to predict "day after tomorrow"
    df['label_2d'] = df['bucket_label'].shift(-2)
    
    df = df.dropna(subset=['lag_1', 'lag_2', 'pct_change_1d', 'label_2d'])
    
    if len(df) < 3:
        st.warning("❗️No hay suficientes datos para generar una predicción a 2 días.")
        return None, None, ['bajo', 'medio', 'alto']

    features = ['lag_1', 'lag_2', 'pct_change_1d', 'day_of_week']
    X = df[features]
    y = df['label_2d']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    # 📌 Use the second-to-last row for features (today = last row, tomorrow = last+1, day after = last+2)

    latest = df.iloc[-2]
    prediction_input = pd.DataFrame([{
        'lag_1': latest['exchange_rate'],
        'lag_2': latest['lag_1'],
        'pct_change_1d': (latest['exchange_rate'] - latest['lag_1']) / latest['lag_1'],
        'day_of_week': (latest['day_of_week'] + 2) % 7  # Day after tomorrow
    }])

    return model, prediction_input, ['bajo', 'medio', 'alto']

# --- function to save prediction ---
def store_prediction(pred_date, predicted):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Add note if skipping the weekend
    note = "fin de semana" if pd.to_datetime(pred_date).weekday() >= 0 and (pd.to_datetime(pred_date) - dt.today()).days > 2 else None

    cursor.execute("""
        INSERT OR REPLACE INTO performance_history 
        (prediction_date, predicted_bucket, actual_bucket, success, note)
        VALUES (?, ?, NULL, NULL, ?)
    """, (pred_date, predicted, note))

    conn.commit()
    conn.close()
    
# ---find the next date with expected exchange data ---    
from datetime import datetime as dt, timedelta

def get_next_valid_prediction_date(from_date):
    # Move 2 days ahead
    target = from_date + timedelta(days=2)
    # Skip weekend days (Saturday = 5, Sunday = 6)
    while target.weekday() >= 5:
        target += timedelta(days=1)
    return target.strftime('%Y-%m-%d')

def get_next_prediction_date_avoiding_duplicates(from_date):
    # Start at +2 days and move forward to next weekday not yet predicted
    target = from_date + timedelta(days=2)
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    while True:
        # Skip weekends
        while target.weekday() >= 5:
            target += timedelta(days=1)

        date_str = target.strftime('%Y-%m-%d')
        cursor.execute("SELECT 1 FROM performance_history WHERE prediction_date = ?", (date_str,))
        exists = cursor.fetchone()
        if not exists:
            conn.close()
            return date_str  # Found a valid date not yet predicted

        target += timedelta(days=1)  # Try next day


# --- Update prediction history ---
def update_performance_history():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Get all prediction dates with no actual
    cursor.execute("""
        SELECT prediction_date, predicted_bucket FROM performance_history
        WHERE actual_bucket IS NULL
    """)
    pending = cursor.fetchall()

    updates = []
    for date_str, predicted in pending:
        formatted_date = pd.to_datetime(date_str).strftime('%Y-%m-%d')
        cursor.execute("""SELECT exchange_rate FROM exchange_rates WHERE end_date=?""", (formatted_date,))        
        row = cursor.fetchone()
        if not row:
            continue  # still no rate for that day

        decimal = row[0] % 1
        if decimal < 0.34:
            actual = 'bajo'
        elif decimal < 0.67:
            actual = 'medio'
        else:
            actual = 'alto'
        success = 1 if actual == predicted else 0
        updates.append((actual, success, date_str))

    if updates:
        cursor.executemany("""
            UPDATE performance_history
            SET actual_bucket = ?, success = ?
            WHERE prediction_date = ?
        """, updates)

    # Calculate success rate
    cursor.execute("SELECT AVG(success) FROM performance_history WHERE success IS NOT NULL")
    kpi = cursor.fetchone()[0] or 0.0

    conn.commit()
    conn.close()
    return round(kpi * 100, 2)  # success rate in percentage


# --- Streamlit App UI ---
st.title("💱 USD-COP Seguimiento de Tasa de Cambio")

if st.button("📥 Actualizar Tasa de Cambio de Hoy"):
    rate, message = fetch_and_store_today_rate()
    if rate is not None:
        st.write(f"💵 Tarifa de Hoy: **{rate:.2f} COP/USD**")
    if "✅" in message:
        st.success(message)
    else:
        st.info(message)

# Show most recent entries
st.subheader("📊 Historial - Tasas de Cambio")
df_latest = load_latest_rates()
st.dataframe(df_latest)

# --- Bucket Prediction Section ---
st.subheader("Pasado Mañana Decimal Bucket Prediction")

# Train the model and get 2 day ahead prediction
model, future_input, labels = train_model()
if model is None or future_input is None:
    st.warning("❗️ No hay suficientes datos para generar una predicción de pasado mañana.")
else:
    probs = model.predict_proba(future_input)[0]
    predicted_label = labels[probs.argmax()]

    # Determine prediction date
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY start_date DESC LIMIT 1", conn)
    conn.close()

    # Use end_date as the true 'valid until' date
    last_valid_date = pd.to_datetime(df["end_date"].iloc[0])
    prediction_date = get_next_valid_prediction_date(last_valid_date)


    # Show log if the predicted date skips the weekend
    day_gap = (pd.to_datetime(prediction_date) - last_valid_date).days
    if day_gap > 2:
        st.info(f"📆 Se saltó el fin de semana. La predicción se generó para el **{prediction_date}**.")


    # Store prediction
    # Check if prediction already exists to avoid duplicates
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM performance_history WHERE prediction_date = ?", (prediction_date,))
    already_exists = cursor.fetchone()
    conn.close()

    if already_exists:
        st.info(f"📌 Ya existe una predicción para el {prediction_date}. No se generó otra.")
    else:
        store_prediction(prediction_date, predicted_label)
    append_performance_row(prediction_date, predicted_label)
    st.success(f"✅ Predicción almacenada para el {prediction_date}.")


    # Show probabilities
    st.markdown(f"📅 Predicción para **{prediction_date}**:")
    for label, prob in zip(labels, probs):
        st.write(f" **{label.upper()}**: {prob:.1%} de probabilidad")
        
        
# --- Prediction History Table ---
st.subheader("Rendimiento Historico")
conn = sqlite3.connect(DB_NAME)
# Calculate cumulative accuracy from performance_history
cursor = conn.cursor()
cursor.execute("""
    SELECT AVG(success) 
    FROM performance_history 
    WHERE success IS NOT NULL
""")
avg_success = cursor.fetchone()[0] or 0.0
conn.close()

# Show metric
st.metric("Precisión acumulada", f"{avg_success*100:.2f}%")

# Show the deduped performance table
conn = sqlite3.connect(DB_NAME)
df_perf = pd.read_sql_query("""
    SELECT prediction_date, predicted_bucket, actual_bucket, success
    FROM performance_history
    ORDER BY prediction_date DESC
""", conn)
conn.close()

st.dataframe(df_perf, height=200)


# --- Line Chart: Daily Decimal Movement (Last 30 Days) ---
st.subheader("📈 Movimiento decimal en el tiempo")

time_range = st.selectbox(
    "Seleccione el periodo de tiempo:",
    options=["Últimos 7 días", "Últimos 30 días", "Últimos 90 días"]
)

days_map = {
    "Últimos 7 días": 7,
    "Últimos 30 días": 30,
    "Últimos 90 días": 90
}

conn = sqlite3.connect(DB_NAME)
df_all = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY start_date", conn)
conn.close()

df_all['start_date'] = pd.to_datetime(df_all['start_date'])
df_all['decimal'] = df_all['exchange_rate'] % 1

# Filter by selected range
days = days_map[time_range]
df_filtered = df_all[df_all['start_date'] >= pd.Timestamp.now() - pd.Timedelta(days=days)]

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_filtered['start_date'], df_filtered['decimal'], marker='o', color='blue', linewidth=2)

# Add horizontal dashed lines for bucket thresholds
ax.axhline(0.34, color='green', linestyle='--', linewidth=1.5, label='Límite grupo 0.34')
ax.axhline(0.67, color='red', linestyle='--', linewidth=1.5, label='Límite grupo 0.67')

# Format x-axis
ax.set_xticks(df_filtered['start_date'])
ax.set_xticklabels(df_filtered['start_date'].dt.strftime('%b %d'), rotation=45, ha='right')

ax.set_title(f"Decimales - {time_range}")
ax.set_xlabel("Fecha")
ax.set_ylabel("Valor Decimal")
ax.grid(True)
ax.set_ylim(0, 1)
ax.legend()

st.pyplot(fig)
