import streamlit as st
import pandas as pd
import google.generativeai as genai
import os

# ----------------------------------
# CONFIG
# ----------------------------------
st.set_page_config(
    page_title="Medical AI Assistant Agent",
    layout="wide"
)

st.title("🩺 Medical AI Assistant Agent")
st.caption("AI-powered clinical decision support for doctors")

# ----------------------------------
# API KEY
# ----------------------------------
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("❌ Gemini API key not found. Add it in Streamlit Secrets.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-1.5-pro")

# ----------------------------------
# SIDEBAR
# ----------------------------------
st.sidebar.header("⚙️ Settings")
language = st.sidebar.selectbox(
    "Report language",
    ["English", "French"]
)

# ----------------------------------
# CSV UPLOAD
# ----------------------------------
st.subheader("📂 Upload Patient Test Results (CSV)")

uploaded_file = st.file_uploader(
    "Upload lab test results CSV",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.success("✅ CSV uploaded successfully")
    st.dataframe(df)

    # ----------------------------------
    # PROMPT
    # ----------------------------------
    prompt = f"""
You are an experienced medical assistant AI supporting licensed doctors.

The following is a patient's medical test results provided as a CSV table.

TASKS:
1. Analyze the lab values carefully.
2. Identify abnormal results.
3. Suggest possible diseases or conditions (differential diagnosis).
4. Recommend possible medications (generic names).
5. Recommend medical procedures or additional tests.
6. Suggest treatment plans and lifestyle advice if applicable.
7. Clearly state uncertainty where appropriate.
8. Provide a structured medical report for a doctor.
9. DO NOT claim to replace a doctor.
10. Use professional medical language.

LANGUAGE: {language}

CSV DATA:
{df.to_csv(index=False)}

OUTPUT FORMAT:
- Summary
- Key Abnormal Findings
- Possible Diagnoses
- Recommended Medications
- Recommended Procedures / Tests
- Treatment & Management Suggestions
- Warnings & Notes for Doctor
- Disclaimer
"""

    # ----------------------------------
    # GENERATE REPORT
    # ----------------------------------
    if st.button("🧠 Analyze & Generate Medical Report"):
        with st.spinner("Analyzing medical data..."):
            response = model.generate_content(prompt)

        st.subheader("📄 AI Medical Report")
        st.markdown(response.text)

        # ----------------------------------
        # DOWNLOAD REPORT
        # ----------------------------------
        st.download_button(
            label="⬇️ Download Report",
            data=response.text,
            file_name="medical_ai_report.txt",
            mime="text/plain"
        )

# ----------------------------------
# FOOTER DISCLAIMER
# ----------------------------------
st.markdown("---")
st.warning("""
⚠️ **Medical Disclaimer**  
This AI assistant provides clinical decision support only.  
It does NOT diagnose diseases and must NOT replace professional medical judgment.
""")
