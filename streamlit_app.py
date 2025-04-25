import streamlit as st
import sqlite3
import requests
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

DB_NAME = "usd_cop_data.db"

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
    df = df.dropna(subset=['lag_1', 'lag_2', 'pct_change_1d'])

    features = ['lag_1', 'lag_2', 'pct_change_1d', 'day_of_week']
    X = df[features]
    y = df['bucket_label']

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    latest = df.iloc[-1]
    tomorrow_features = pd.DataFrame([{
        'lag_1': latest['exchange_rate'],
        'lag_2': latest['lag_1'],
        'pct_change_1d': (latest['exchange_rate'] - latest['lag_1']) / latest['lag_1'],
        'day_of_week': (latest['day_of_week'] + 1) % 7
    }])

    return model, tomorrow_features, ['bajo', 'medio', 'alto']


# --- Streamlit App UI ---
st.title("🇺🇸💱 USD-COP Daily Rate Tracker")

if st.button("📥 Update Today's USD-COP Rate"):
    rate, message = fetch_and_store_today_rate()
    if rate is not None:
        st.write(f"💵 Today's Rate: **{rate:.2f} COP/USD**")
    st.success(message) if "✅" in message else st.info(message)

# Show most recent entries
st.subheader("📊 Recent Exchange Rates")
df_latest = load_latest_rates()
st.dataframe(df_latest)

# --- Bucket Prediction Section ---
st.subheader("Tomorrow's Decimal Bucket Prediction")

# Train the model and get prediction
model, tomorrow_X, labels = train_model()
probs = model.predict_proba(tomorrow_X)[0]

# Display probabilities
for label, prob in zip(labels, probs):
    st.write(f" **{label.upper()}**: {prob:.1%} chance")
    


# --- Bucket Trend Over Time (Last 3 Months) ---
st.subheader("📊 Weekly Bucket Distribution - Last 3 Months")

# Reload data
conn = sqlite3.connect(DB_NAME)
df_all = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY start_date", conn)
conn.close()

df_all['start_date'] = pd.to_datetime(df_all['start_date'])
df_all['decimal'] = df_all['exchange_rate'] % 1

def bucketize(decimal):
    if decimal < 0.34:
        return 'bajo'
    elif decimal < 0.67:
        return 'medio'
    else:
        return 'alto'

df_all['bucket'] = df_all['decimal'].apply(bucketize)

# Filter last 3 months
df_recent = df_all[df_all['start_date'] >= pd.Timestamp.now() - pd.DateOffset(months=3)]

# Create week column
df_recent['week'] = df_recent['start_date'].dt.to_period('W').apply(lambda r: r.start_time)

# Count buckets per week
weekly_counts = df_recent.groupby(['week', 'bucket']).size().unstack(fill_value=0)

# Plot
weekly_counts = weekly_counts.sort_index()
fig, ax = plt.subplots(figsize=(10, 5))
weekly_counts.plot(kind='bar', stacked=True, ax=ax, color={'bajo': 'blue', 'medio': 'orange', 'alto': 'red'})

ax.set_title("Decimal Bucket Distribution per Week (Last 3 Months)")
ax.set_ylabel("Count of Days")
ax.set_xticklabels(weekly_counts.index.strftime('%Y-%m-%d'), rotation=45, ha='right')
ax.set_xlabel("Week Starting")
ax.legend(title="Bucket")

st.pyplot(fig)

# --- Line Chart: Daily Decimal Movement (Last 30 Days) ---
st.subheader("📉 Decimal Movement Over Time")

# Time range selector
time_range = st.selectbox(
    "Select time period:",
    options=["Last 7 days", "Last 30 days", "Last 90 days"]
)

# Map selection to days
days_map = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90
}

# Load and prep data
conn = sqlite3.connect(DB_NAME)
df_all = pd.read_sql_query("SELECT * FROM exchange_rates ORDER BY start_date", conn)
conn.close()

df_all['start_date'] = pd.to_datetime(df_all['start_date'])
df_all['decimal'] = df_all['exchange_rate'] % 1

# Filter based on selected range
days = days_map[time_range]
df_filtered = df_all[df_all['start_date'] >= pd.Timestamp.now() - pd.Timedelta(days=days)]

# Plot
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df_filtered['start_date'], df_filtered['decimal'], marker='o', color='blue', linewidth=2)

# Format x-axis: show just day and month
ax.set_xticks(df_filtered['start_date'])
ax.set_xticklabels(df_filtered['start_date'].dt.strftime('%b %d'), rotation=45, ha='right')

ax.set_title(f"Decimal Values - {time_range}")
ax.set_xlabel("Date")
ax.set_ylabel("Decimal Value")
ax.grid(True)
ax.set_ylim(0, 1)
st.pyplot(fig)
