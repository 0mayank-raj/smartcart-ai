import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="SmartCart AI", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.card {
    background-color: #1E1E2F;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 15px;
}
.highlight {
    color: #00FFAA;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------- HEADER ----------------
st.title("🛒 SmartCart AI Dashboard")
st.markdown("Analyze customer behavior and generate actionable insights")

# ---------------- INPUT SECTION ----------------
st.markdown("## 🧾 Enter Customer Details")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", 10, 100)

with col2:
    income = st.number_input("Annual Income (₹)")

with col3:
    spending = st.number_input("Total Spending (₹)")

# ---------------- BUTTON ----------------
if st.button("🚀 Generate Dashboard"):

    input_data = np.array([[age, income, spending]])
    input_scaled = scaler.transform(input_data)
    cluster = model.predict(input_scaled)[0]

    st.markdown("---")

    # ---------------- SEGMENT ----------------
    st.markdown("## 🎯 Customer Segment")

    if cluster == 0:
        segment = "💰 Budget Customer"
        desc = "Prefers low-cost products and discounts."
    elif cluster == 1:
        segment = "🔥 Premium Customer"
        desc = "High-value customer who spends aggressively."
    else:
        segment = "🧠 Balanced Customer"
        desc = "Moderate spending with growth potential."

    st.markdown(f"""
    <div class="card">
        <h3 class="highlight">{segment}</h3>
        <p>{desc}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- METRICS ----------------
    st.markdown("## 📊 Key Metrics")

    ratio = spending / income if income != 0 else 0

    col1, col2, col3 = st.columns(3)

    col1.metric("Age", age)
    col2.metric("Income (₹)", f"{income:,.0f}")
    col3.metric("Spending (₹)", f"{spending:,.0f}")

    st.metric("Spending Ratio", f"{ratio:.2f}")

    # ---------------- INSIGHTS ----------------
    st.markdown("## 💡 Insights")

    if ratio > 0.6:
        spending_msg = "⚠️ Very high spender"
    elif ratio > 0.3:
        spending_msg = "👍 Moderate spender"
    else:
        spending_msg = "💡 Conservative spender"

    age_msg = "🎯 Young and active buyer" if age < 30 else "📊 Mature customer"

    st.markdown(f"""
    <div class="card">
        <p>{spending_msg}</p>
        <p>{age_msg}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- RECOMMENDATIONS ----------------
    st.markdown("## 📢 Business Strategy")

    if cluster == 0:
        strategy = "Offer discounts, cashback, and budget deals."
    elif cluster == 1:
        strategy = "Promote premium products and exclusive offers."
    else:
        strategy = "Use combo deals and upselling techniques."

    st.markdown(f"""
    <div class="card">
        <p>{strategy}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- VISUALIZATION ----------------
    st.markdown("## 📈 Customer Overview")

    data = pd.DataFrame({
        "Feature": ["Income", "Spending"],
        "Value": [income, spending]
    })

    fig, ax = plt.subplots()
    ax.bar(data["Feature"], data["Value"])
    ax.set_title("Income vs Spending")

    st.pyplot(fig)

    # ---------------- SUMMARY ----------------
    st.markdown("## 🧾 Final Summary")

    st.markdown(f"""
    <div class="card">
    This customer falls under <span class="highlight">{segment}</span>.  
    Their spending behavior suggests: <b>{spending_msg}</b>.  
    Recommended action: <b>{strategy}</b>
    </div>
    """, unsafe_allow_html=True)