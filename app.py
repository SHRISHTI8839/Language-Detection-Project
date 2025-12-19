import streamlit as st
import pickle

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Language Detection Pro",
    page_icon="🌍",
    layout="wide"
)

# ================= LOAD MODEL =================
cv = pickle.load(open("cv.pkl", "rb"))
model = pickle.load(open("language_model.pkl", "rb"))

# ================= CUSTOM CSS =================
st.markdown("""
<style>
.main {
    background-color: #f8fafc;
}
.title-text {
    font-size: 40px;
    font-weight: 700;
}
.subtitle-text {
    font-size: 18px;
    color: #555;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 12px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
}
.result-box {
    background-color: #e6fffa;
    padding: 20px;
    border-left: 6px solid #14b8a6;
    border-radius: 8px;
    font-size: 20px;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
with st.sidebar:
    st.image(
        "https://cdn-icons-png.freepik.com/512/10655/10655641.png",
        width=120
    )
    st.markdown("### ⚙️ Settings & Info")
    st.info("This AI model can detect **15+ languages** with high accuracy.")
    st.markdown("---")
    st.markdown("👩‍💻 **Developed by:** Shrishti Singh")
    st.markdown("👩‍💻 **Developed by:** Siddhant Singh")
    st.markdown("👩‍💻 **Developed by:** Greshee Sinha")
    st.markdown("👩‍💻 **Developed by:** Abhay Tripathi")
    st.markdown("🎓 **Mini Project | Machine Learning**")

# ================= MAIN UI =================
st.markdown("<div class='title-text'>🌍 Language Detection Pro</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle-text'>Real-time Language Identification using Machine Learning</div>",
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns([1.2, 1])

# ========== LEFT COLUMN ==========
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    user_text = st.text_area(
        "✍️ Enter text below",
        height=180,
        placeholder="Example: Bonjour tout le monde / Hello how are you / नमस्ते"
    )

    detect_btn = st.button("🚀 Identify Language", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ========== RIGHT COLUMN ==========
with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### 📊 Detection Result")

    if detect_btn:
        if user_text.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            data = cv.transform([user_text]).toarray()
            prediction = model.predict(data)[0]

            st.markdown(
                f"<div class='result-box'>Detected Language: <br>🌐 <b>{prediction}</b></div>",
                unsafe_allow_html=True
            )
    else:
        st.info("ℹ️ Enter text and click the button to detect language.")

    st.markdown("</div>", unsafe_allow_html=True)

# ================= FOOTER =================
st.markdown("<div class='footer'>© 2025 Language Detection Pro | Mini Project</div>", unsafe_allow_html=True)








