import os
import re
import csv
import logging
import pdfplumber
import pandas as pd
import streamlit as st
from twilio.rest import Client
from crewai import Agent, Task, Crew

# Set logging level
logging.getLogger("pdfminer").setLevel(logging.ERROR)
# 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
# API keys
os.environ["OPENAI_API_KEY"] = "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"  # Use environment variable or st.secrets
TWILIO_ACCOUNT_SID = 'ACce4a2954135b0af9f6be1b9cb2d097ad'
TWILIO_AUTH_TOKEN = '3f579afcf08ef1ab69c43f91e44640a0'
TWILIO_PHONE_NUMBER = '+19514740658'
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

# Paths
verified_csv_path = "C:/Generative Ai/UniversityHelpdeskAutomation/Documents/verified_students.csv"
final_csv_path = "C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv"

# Admission Agent
admission_agent = Agent(
    role="Admission Assistant",
    goal="Guide students through the admission process",
    backstory="You are the official assistant of NovaTech University. Welcome students and explain required documents.",
    verbose=False,
    allow_delegation=False
)

intro_task = Task(
    description="Create a welcome message explaining required documents: Birth Certificate, Aadhar Card, 12th Marksheet, Rank Card, and Domicile Certificate.",
    expected_output="Welcome to NovaTech! Please upload the following documents: Birth Certificate, Aadhar Card, 12th Marksheet, Rank Card, and Domicile Certificate to begin your admission process.",
    agent=admission_agent
)

# Extraction Agents
document_agent = Agent(
    role="Document Extractor",
    goal="Extract student information from official documents",
    backstory="You identify name, DOB, Aadhaar number, rank, domicile validity, marks %, and document verification.",
    verbose=False,
    allow_delegation=False
)

verifier_agent = Agent(
    role="Data Verifier",
    goal="Verify extracted student data and report inconsistencies",
    backstory="You ensure correctness and formatting of student info including Aadhaar and DOB consistency.",
    verbose=False,
    allow_delegation=False
)

# Helpers
def extract_text_from_pdf(path):
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() for p in pdf.pages if p.extract_text())

def extract_phone_number(text):
    match = re.search(r'\b\d{10}\b|\b\d{3}[- ]?\d{3}[- ]?\d{4}\b', text)
    return str(re.sub(r'[- ]', '', match.group())) if match else "Not Found"

def format_aadhar_for_csv(aadhar_str):
    digits = re.sub(r'\D', '', aadhar_str)
    return f"{digits[:4]} {digits[4:8]} {digits[8:]}" if len(digits) == 12 else aadhar_str

def extract_all_aadhar_numbers(text):
    return re.findall(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", text)

def extract_domicile_section(text):
    start = text.lower().find("domicile")
    end = text.lower().find("marksheet")
    return text[start:end] if end > start else text[start:]

def clean_percentage(val): return float(str(val).replace('%', '').strip()) if val else 0.0
def clean_rank(val): return int(str(val).strip()) if val and str(val).isdigit() else float('inf')

ELIGIBILITY_CRITERIA = {
    "base": {"min_rank": 20000, "min_percentage": 50.0, "require_domicile": True, "require_document_verified": True},
    "branches": {
        "CSE": {"rank": 4000, "percentage": 95.0},
        "IT": {"rank": 6000, "percentage": 80.0},
        "ECE": {"rank": 10000, "percentage": 85.0},
        "EE": {"rank": 15000, "percentage": 70.0},
        "ME": {"rank": 20000, "percentage": 50.0}
    }
}

def determine_eligible_streams(row):
    if row['DomicileValidity'] == 'Approved' and row['Document verified'].lower() == 'yes' and row['Rank'] < 20000 and row['12 percentage'] > 50.0:
        return ", ".join([b for b, c in ELIGIBILITY_CRITERIA['branches'].items() if row['Rank'] <= c['rank'] and row['12 percentage'] >= c['percentage']]) or "No"
    return "No"

def send_sms(to, name, streams):
    try:
        msg = client.messages.create(
            body=f"Hi {name}, Congrats! 🎉 You're eligible for: {streams}. Visit admission portal.",
            from_=TWILIO_PHONE_NUMBER,
            to=to
        )
        return "Sent"
    except Exception as e:
        return f"Error: {e}"

# --- Streamlit Frontend ---
st.set_page_config(page_title="NovaTech Admission Portal", layout="centered")

# Intro message
intro_crew = Crew(agents=[admission_agent], tasks=[intro_task])
with st.expander("📢 Admission Info", expanded=True):
    st.success(intro_crew.kickoff())

uploaded_files = st.file_uploader("📄 Upload student PDFs", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_student_data = {}
    st.info("⏳ Processing uploaded documents...")

    for uploaded_file in uploaded_files:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.read())

        try:
            text = extract_text_from_pdf(uploaded_file.name)
            phone_number = extract_phone_number(text)

            task = Task(
                description=(
                    "Extract the following:\n"
                    "- Name\n- DOB\n- Aadhar No.\n- Rank\n- Domicile Validity\n"
                    "- 12 Percentage Marks %\n- Document Verified\n\n"
                    f"Content:\n{text}"
                ),
                expected_output=(
                    "Name: <name>\nDOB: <date>\nAadhar No.: <number>\n"
                    "Rank: <rank>\nDomicile Validity: <status>\n"
                    "12 Percentage Marks %: <percent>\nDocument Verified: Yes"
                ),
                agent=document_agent
            )
            crew = Crew(agents=[document_agent], tasks=[task])
            result = str(crew.kickoff())

            name_line = next(line for line in result.splitlines() if "Name:" in line)
            student_name = name_line.split(":", 1)[1].strip()

            all_student_data[student_name] = {
                "extracted_info": result,
                "raw_text": text,
                "phone_number": phone_number
            }

        except Exception as e:
            all_student_data[f"Unknown_{uploaded_file.name}"] = {"error": str(e)}

    # Verification
    verified_rows = []
    for student, data in all_student_data.items():
        if "error" in data:
            continue

        extracted_info = data["extracted_info"]
        full_text = data["raw_text"]

        verify_task = Task(
            description=(
                f"Student info:\n{extracted_info}\n\nFull content:\n{full_text}\n"
                "Check:\n1. Name format\n2. Aadhar validity\n3. Name consistency\n"
                "4. DOB consistency\n5. Format DOB like 16 July 2004"
            ),
            expected_output=(
                "Validation Report:\n- Name Validity: ✅ Pass/❌ Fail\n"
                "- Aadhar Validity: ✅ Pass/❌ Fail\n- Name Consistency: ✅ Pass/❌ Fail\n"
                "- DOB Consistency: ✅ Pass/❌ Fail\n- DOB Format (English): <formatted>\n\n"
                "*Detailed Explanation:* ..."
            ),
            agent=verifier_agent
        )

        crew = Crew(agents=[verifier_agent], tasks=[verify_task])
        validation_result = str(crew.kickoff())

        info = {}
        for line in data["extracted_info"].splitlines():
            if ":" in line:
                key, val = line.split(":", 1)
                info[key.strip()] = val.strip()

        raw_aadhar = info.get("Aadhar No.", "")
        aadhar_digits = re.sub(r'\D', '', raw_aadhar)
        domicile_section = extract_domicile_section(data["raw_text"])
        found_aadhars = [re.sub(r'\D', '', x) for x in extract_all_aadhar_numbers(domicile_section)]
        force_fail = aadhar_digits not in found_aadhars
        dom_validity = "Not Approved" if force_fail else info.get("Domicile Validity", "")

        passed_checks = all(kw in validation_result for kw in [
            "- Name Validity: ✅ Pass", "- Aadhar Validity: ✅ Pass",
            "- Name Consistency: ✅ Pass", "- DOB Consistency: ✅ Pass"
        ])
        doc_verified = "yes" if passed_checks and not force_fail else "no"

        marks = info.get("12 Percentage Marks %", "")
        if marks and not marks.endswith("%"):
            marks += "%"

        row = {
            "Name": info.get("Name", ""),
            "Phone Number": data.get("phone_number", ""),
            "DOB": info.get("DOB", ""),
            "Aadhar No.": format_aadhar_for_csv(raw_aadhar),
            "Rank": clean_rank(info.get("Rank", "")),
            "DomicileValidity": dom_validity,
            "12 percentage": clean_percentage(marks),
            "Document verified": doc_verified
        }

        row["Eligible Streams"] = determine_eligible_streams(row)
        verified_rows.append(row)

    df = pd.DataFrame(verified_rows)
    df.to_csv(verified_csv_path, index=False)
    df.to_csv(final_csv_path, index=False)

    st.success("✅ Extraction & verification complete!")
    st.dataframe(df)

    if st.button("📲 Send SMS to eligible students"):
        for _, row in df.iterrows():
            if row["Eligible Streams"] != "No":
                phone_digits = re.sub(r'\D', '', row["Phone Number"])
                full_number = '+91' + phone_digits if len(phone_digits) == 10 else ''
                if full_number:
                    send_sms(full_number, row["Name"], row["Eligible Streams"])
        st.success("🎉 SMS sent to eligible students!")
        

    import subprocess

    st.markdown("---")
    if st.button("🚀 Run Next Process"):
        try:
            result = subprocess.run(["python", "app.py"], capture_output=True, text=True)
            st.success("✅ Next process executed successfully!")
            st.code(result.stdout)  # Optionally display the output
        except Exception as e:
            st.error(f"❌ Failed to run the script: {e}")
