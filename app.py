import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import plotly.express as px

# ---------------- LOAD MODEL ----------------
model = load_model("model.keras", compile=False)

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("cat_encoder.pkl", "rb") as f:
    cat_encoder = pickle.load(f)

with open("urg_encoder.pkl", "rb") as f:
    urg_encoder = pickle.load(f)

max_len = 120

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Support System", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title {
    font-size:38px;
    font-weight:700;
    text-align:center;
    color:#1f2c3b;
}

.subtitle {
    text-align:center;
    color:#6c757d;
    margin-bottom:25px;
}

.card {
    background-color:#f8f9fa;
    padding:20px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.05);
    margin-bottom:15px;
}

.result-box {
    padding:15px;
    border-radius:10px;
    background:#ffffff;
    border-left:5px solid #4CAF50;
    margin-bottom:10px;
}

.small-text {
    color:#6c757d;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-title">AI Customer Support System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Based Ticket Classification & Urgency Detection</div>', unsafe_allow_html=True)

# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- INPUT ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)

user_input = st.text_area("Enter Customer Issue", height=120)

predict_btn = st.button("Analyze Ticket")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if predict_btn:

    if user_input.strip() == "":
        st.warning("Please enter valid input.")
    else:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=max_len)

        pred_cat, pred_urg = model.predict(padded)

        # CATEGORY
        cat_index = np.argmax(pred_cat)
        cat_label = cat_encoder.inverse_transform([cat_index])[0]
        cat_conf = float(np.max(pred_cat))

        # URGENCY
        urg_index = np.argmax(pred_urg)
        urg_label = urg_encoder.inverse_transform([urg_index])[0]
        urg_conf = float(np.max(pred_urg))

        # ---------------- RESULTS ----------------
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Category Prediction")

            st.markdown(f"""
            <div class="result-box">
            <b>{cat_label}</b><br>
            <span class="small-text">Confidence: {cat_conf:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            fig_cat = px.pie(
                values=pred_cat[0],
                names=cat_encoder.classes_,
                title="Category Distribution"
            )
            st.plotly_chart(fig_cat, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Urgency Prediction")

            st.markdown(f"""
            <div class="result-box">
            <b>{urg_label}</b><br>
            <span class="small-text">Confidence: {urg_conf:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

            fig_urg = px.pie(
                values=pred_urg[0],
                names=urg_encoder.classes_,
                title="Urgency Distribution"
            )
            st.plotly_chart(fig_urg, use_container_width=True)

            st.markdown('</div>', unsafe_allow_html=True)

        # ---------------- HISTORY ----------------
        st.session_state.history.append({
            "input": user_input,
            "category": cat_label,
            "urgency": urg_label,
            "confidence": cat_conf
        })

# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("Prediction History")

for item in reversed(st.session_state.history):
    st.markdown(f"""
    <div class="card">
    <b>Issue:</b> {item['input']} <br><br>
    <b>Category:</b> {item['category']} <br>
    <b>Urgency:</b> {item['urgency']} <br>
    <b>Confidence:</b> {item['confidence']:.2f}
    </div>
    """, unsafe_allow_html=True)