import streamlit as st
import pandas as pd
import google.generativeai as genai

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="🧬 AI Medical Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI Medical Assistant")
st.markdown(
    "This AI agent assists doctors in interpreting lab results, "
    "suggesting possible diagnoses, and recommending next clinical steps."
)

# -------------------------------------------------
# GEMINI API KEY (Streamlit Secrets)
# -------------------------------------------------
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ Gemini API key not found. Add GOOGLE_API_KEY in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# -------------------------------------------------
# ✅ BACKWARD-COMPATIBLE MODEL (IMPORTANT)
# -------------------------------------------------
model = genai.GenerativeModel("gemini-pro")

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "patient_data" not in st.session_state:
    st.session_state.patient_data = ""

# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------
st.subheader("📂 Upload Patient Lab Results")

uploaded_file = st.file_uploader(
    "Upload patient results (CSV or TXT)",
    type=["csv", "txt"]
)

if uploaded_file:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.dataframe(df)
        patient_text = df.to_csv(index=False)
    else:
        patient_text = uploaded_file.read().decode("utf-8")

    st.session_state.patient_data = patient_text
    st.success("✅ File uploaded successfully")

    extra_context = st.text_area(
        "📝 Additional clinical context (symptoms, history, age, sex, etc.)"
    )

    # -------------------------------------------------
    # MAIN ANALYSIS
    # -------------------------------------------------
    if st.button("🧠 Interpret & Generate Medical Report"):
        with st.spinner("Analyzing medical data..."):

            prompt = f"""
You are a senior medical AI assistant supporting licensed physicians.

PATIENT LAB RESULTS:
{patient_text}

CLINICAL CONTEXT:
{extra_context}

TASKS:
1. Interpret the lab results clearly.
2. Identify abnormal findings.
3. Provide possible diagnoses (differential diagnosis).
4. Suggest additional tests or procedures.
5. Recommend possible treatments or medications (generic names).
6. Highlight risks and uncertainty.
7. Produce a structured medical report.

RULES:
- You do NOT replace a doctor.
- Use professional medical language.
- Add a medical disclaimer at the end.
"""

            response = model.generate_content(prompt)
            ai_reply = response.text

            st.session_state.chat_history = [
                {"role": "assistant", "content": ai_reply}
            ]

            st.subheader("📄 AI Medical Report")
            st.markdown(ai_reply)

# -------------------------------------------------
# FOLLOW-UP CHAT
# -------------------------------------------------
if st.session_state.patient_data:
    st.markdown("### 💬 Ask Follow-up Medical Questions")

    follow_up = st.text_input(
        "Ask a follow-up question based on the uploaded results"
    )

    if follow_up:
        st.session_state.chat_history.append(
            {"role": "user", "content": follow_up}
        )

        with st.spinner("AI is thinking..."):

            chat_prompt = f"""
You are a medical AI assistant continuing a discussion with a doctor.

PATIENT DATA:
{st.session_state.patient_data}

CONVERSATION HISTORY:
{st.session_state.chat_history}

Doctor's question:
{follow_up}
"""

            response = model.generate_content(chat_prompt)
            ai_reply = response.text

            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_reply}
            )

            st.markdown("**🤖 AI Response:**")
            st.markdown(ai_reply)

# -------------------------------------------------
# CHAT HISTORY
# -------------------------------------------------
with st.expander("📝 Conversation History"):
    for msg in st.session_state.chat_history:
        speaker = "👨‍⚕️ Doctor" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{speaker}:** {msg['content']}")

# -------------------------------------------------
# DISCLAIMER
# -------------------------------------------------
st.markdown("---")
st.warning(
    "⚠️ This AI assistant provides clinical decision support only. "
    "It does NOT diagnose diseases and must NOT replace professional medical judgment."
)
