import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ----------------------------------
# PAGE CONFIG
# ----------------------------------
st.set_page_config(
    page_title="🧬 AI Medical Assistant",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 AI Medical Assistant")
st.markdown(
    "This AI agent assists doctors in interpreting lab results, "
    "suggesting possible diagnoses, and recommending next steps."
)

# ----------------------------------
# GEMINI API KEY (Streamlit Secrets)
# ----------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Please add it to Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# ✅ STABLE MODEL (IMPORTANT)
model = genai.GenerativeModel("models/gemini-1.0-pro")

# ----------------------------------
# SESSION STATE
# ----------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ----------------------------------
# FILE UPLOAD
# ----------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload patient results (CSV or TXT)",
    type=["csv", "txt"]
)

result_text = None

if uploaded_file:
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        st.subheader("📊 Uploaded Lab Results")
        st.dataframe(df)
        result_text = df.to_csv(index=False)
    else:
        result_text = uploaded_file.read().decode("utf-8")

    st.success("✅ File uploaded successfully")

    extra_context = st.text_area(
        "📝 Additional clinical context (symptoms, history, age, sex, etc.)"
    )

    if st.button("🧠 Interpret & Predict Disease"):
        with st.spinner("Analyzing medical data..."):

            prompt = f"""
You are a senior medical AI assistant supporting licensed physicians.

PATIENT LAB RESULTS:
{result_text}

CLINICAL CONTEXT:
{extra_context}

TASKS:
1. Interpret the lab results.
2. Highlight abnormal findings.
3. Suggest possible diagnoses (differential diagnosis).
4. Recommend further tests or procedures.
5. Suggest possible treatments or medications (generic names).
6. Mention uncertainty and risks.
7. Produce a clear report for a doctor.

IMPORTANT:
- Do NOT claim to replace a doctor.
- Use professional medical language.
- Add a medical disclaimer at the end.
"""

            response = model.generate_content(prompt)
            ai_reply = response.text

            st.session_state.chat_history.append(
                {"role": "assistant", "content": ai_reply}
            )

            st.subheader("📄 AI Medical Interpretation")
            st.markdown(ai_reply)

# ----------------------------------
# FOLLOW-UP CHAT
# ----------------------------------
st.markdown("### 💬 Ask Follow-up Questions")

follow_up = st.text_input(
    "Ask a follow-up question based on the uploaded results..."
)

if follow_up and result_text:
    st.session_state.chat_history.append(
        {"role": "user", "content": follow_up}
    )

    with st.spinner("AI is thinking..."):

        chat_prompt = f"""
You are a medical AI assistant continuing a discussion with a doctor.

PATIENT DATA:
{result_text}

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

        st.markdown("**🤖 AI Answer:**")
        st.markdown(ai_reply)

# ----------------------------------
# CHAT HISTORY
# ----------------------------------
with st.expander("📝 Conversation History"):
    for msg in st.session_state.chat_history:
        speaker = "👨‍⚕️ Doctor" if msg["role"] == "user" else "🤖 AI"
        st.markdown(f"**{speaker}:** {msg['content']}")

# ----------------------------------
# DISCLAIMER
# ----------------------------------
st.markdown("---")
st.warning(
    "⚠️ This AI assistant provides clinical decision support only. "
    "It does NOT diagnose diseases and must NOT replace professional medical judgment."
)
