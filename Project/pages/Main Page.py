import re
import os
import pandas as pd
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from crewai import Agent, Process, Task, Crew

# Set up OpenAI API key
OPENAI_API_KEY = "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"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load and clean PDF
@st.cache_resource
def load_and_process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    for page in pages:
        page.page_content = re.sub(r'==start of OCR for page \d+==', '', page.page_content)
        page.page_content = re.sub(r'==end of OCR for page \d+==', '', page.page_content)
        page.page_content = re.sub(r'\n+', '\n', page.page_content).strip()
    return pages

# Create the QA chain
@st.cache_resource
def create_qa_chain(_pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(_pages)
    embeddings = OpenAIEmbeddings()

    persist_directory = "chroma_db"  # Any folder name you like
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    db.persist()

    retriever = db.as_retriever()
    llm = OpenAI(temperature=0)

    prompt_template = """
    First greet the user whoever says hey or hello to you. Then use the following pieces of context to answer the question at the end.
    Keep in mind that the student is still yet to be admitted in any stream. So after telling everything, advise them to take decision correctly.
    If you don't know the answer, just say you don't know and tell them to contact the authority at +91-98783676 or email novatech.edu.in. Don't make up an answer.

    {context}

    Question: {question}
    Answer:
    """
    prompt = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# Streamlit App UI
st.set_page_config(page_title="NovaTech Chatbot", layout="wide")
st.title("📘 CounselNova")

# Sidebar for navigation
page = st.sidebar.radio("Select Page", ["Chatbot", "Stream Advisor", "Loan","Final"])

# PDF Loading and QA Chain Initialization
file_path = "../NovaTech Institute of Engineering.pdf"
pages = load_and_process_pdf(file_path)
qa_chain = create_qa_chain(pages)

# ------------------ Chatbot Page ------------------
if page == "Chatbot":
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask me anything about NovaTech...")
    if user_input:
        response = qa_chain.run(user_input)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", response))

    for sender, message in st.session_state.chat_history:
        with st.chat_message(sender if sender == "You" else "assistant"):
            st.markdown(message)

# ------------------ Stream Advisor Page ------------------
elif page == "Stream Advisor":
    df = pd.read_csv('C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv')
    df.loc[df["Eligible Streams"].str.strip().str.lower() == "no", "Final Stream"] = "No"
    df.to_csv("C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv", index=False)

    def get_student_data(name: str) -> dict:
        row = df[df["Name"] == name]
        if row.empty:
            return {"error": "Student not found"}
        streams = row["Eligible Streams"].values[0]
        return {
            "Name": name,
            "Eligible Streams": streams,
            "Document verified": row["Document verified"].values[0],
            "Eligible": streams.lower() != "no"
        }

    def is_stream_allowed(name: str, preferred_stream: str) -> bool:
        data = get_student_data(name)
        if "error" in data or not data["Eligible"]:
            return False
        allowed_streams = [s.strip().lower() for s in data["Eligible Streams"].split(',')]
        return preferred_stream.lower() in allowed_streams

    def update_final_stream(name: str, final_stream: str) -> str:
        global df
        idx = df[df["Name"] == name].index
        if len(idx) == 0:
            return "Student not found."
        row = df[df["Name"] == name]
        if final_stream.lower() == "no" or row["Eligible Streams"].values[0].strip().lower() == "no":
            df.loc[idx, "Final Stream"] = "No"
        elif final_stream.lower() not in row["Eligible Streams"].values[0].lower():
            df.loc[idx, "Final Stream"] = "No"
        else:
            df.loc[idx, "Final Stream"] = final_stream
        df.to_csv("C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv", index=False)
        return f"✅ Final Stream for {name} updated to: {df.loc[idx, 'Final Stream'].values[0]}"

    st.title("Stream Advisor")
    student_name = st.text_input("Enter student name:")

    if student_name:
        student_info = get_student_data(student_name)

        if "error" in student_info:
            st.error(f"❌ {student_name} not found.")
        else:
            eligible_streams = student_info["Eligible Streams"]
            st.write(f"📚 Student: {student_name}")
            st.write(f"Eligible Streams: {eligible_streams}")
            st.write(f"Document Verified: {student_info['Document verified']}")

            if eligible_streams.lower() == "no":
                st.warning(f"❌ {student_name} is not eligible for any stream.")
                st.write(update_final_stream(student_name, "No"))
            else:
                preferred_stream = st.selectbox(
                    f"Select your preferred stream from eligible options:",
                    options=[s.strip() for s in eligible_streams.split(',')] + ["No"]
                )

                if st.button("Update Final Stream"):
                    if preferred_stream.lower() == "no":
                        st.warning(f"❌ {student_name} chose not to opt for any stream.")
                        st.write(update_final_stream(student_name, "No"))
                    elif is_stream_allowed(student_name, preferred_stream):
                        st.success(f"✅ {student_name} chose the stream: {preferred_stream}")
                        st.write(update_final_stream(student_name, preferred_stream))
                    else:
                        st.error(f"⚠️ '{preferred_stream}' is not eligible. Please try again.")
    else:
        st.write("Enter a student name to start.")

import pandas as pd
import streamlit as st
from tabulate import tabulate

# ---------------------
# Load CSV
# ---------------------
# verified_csv_path = "C:/Generative Ai/Student Counsellor/Documents/verified_students.csv"
# final_csv_path = "C:/Generative Ai/Student Counsellor/Documents/students_with_eligibility.csv"

df = pd.read_csv('C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv')
print(df)

# ---------------------
# Functional Utilities
# ---------------------
# def get_student_data(name: str) -> dict:
#     """Fetch a student's data by name."""
#     row = df[df["Name"] == name]
#     if not row.empty and row.iloc[0]["Final_Stream"][0] != "No":
#         return {
#             "Name": name,
#             "Rank": int(row["Rank"].values[0]),
#             "12_percentage": float(row["12_percentage"].values[0]),
#             "Final_Stream": row["Final_Stream"].values[0]
#         }
#     else:
#         return {"error": "Student not found"}

def get_student_data(name: str) -> dict:
    if "Name" not in df.columns or "Eligible Streams" not in df.columns:
        print("Eligible Streams" not in df.columns)
        raise ValueError("Required columns missing from the dataset.")
    
    row = df[df["Name"] == name]
    if row.empty:
        return {"error": "Student not found"}
    
    streams = row["Eligible Streams"].values[0]
    return {
        "Name": name,
        "Eligible Streams": streams,
        "Document verified": row["Document verified"].values[0],
        "Eligible": streams.lower() != "no"
    }

def update_loan_eligibility(name: str, rank: int, twelfth_per: float, family_income: float) -> str:
    """Update the loan eligibility status for a student."""
    global df
    idx = df[df["Name"] == name].index
    if len(idx) == 0:
        return "Student not found."

    status = "Yes" if rank < 4000 and twelfth_per > 85 and family_income < 400000 else "No"

    # Add column if not exists
    if "Loan Eligibility" not in df.columns:
        df["Loan Eligibility"] = "Not Checked"

    df.loc[idx, "Loan Eligibility"] = status
    df.to_csv('C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv', index=False)
    return f"Loan eligibility for {name} updated to '{status}'"

# ---------------------
# Streamlit UI Setup
# ---------------------
# st.title("Loan Application")

# Sidebar for loan application page
# page = st.sidebar.radio("Select a Page", ["Home", "Loan"])

if page == "Loan":
    student_name = st.text_input("Enter student name:")
    wants_loan = st.radio("Do you want to apply for an educational loan?", ["Yes", "No"])

    if student_name and wants_loan == "Yes":
        student_info = get_student_data(student_name)
        if "error" in student_info:
            st.error("❌ Student not eligible.")
        else:
            try:
                family_income = st.number_input("Enter family annual income:", min_value=0)
                rank = st.number_input("Enter student rank:", min_value=1)
                twelfth_per = st.number_input("Enter 12th grade percentage:", min_value=0, max_value=100)

                if st.button("Check Loan Eligibility"):
                    result = update_loan_eligibility(student_name, rank, twelfth_per, family_income)
                    st.success(result)

                    # If eligible for loan, calculate the payment schedule
                    if result.startswith("Loan eligibility for") and "Yes" in result:
                        # Normalize column names in df
                        # df.columns = df.columns.str.strip().str.replace(" ", "_")

                        # Check if the student name exists in the dataframe
                        if student_name in df["Name"].values:
                            student_row = df[df["Name"] == student_name].iloc[0]
                            # Now, continue with the existing code
                            stream = student_row["Final Stream"]

                            # Load the Loan Dataset
                            # C:\Generative Ai\Student Counsellor\Documents\students_with_eligibility.csv
                            loan_df = pd.read_csv("C:/Generative Ai/UniversityHelpdeskAutomation/Documents/Loan_Dataset.csv")
                            # loan_df.columns = loan_df.columns.str.strip().str.replace(" ", "_")  # Normalize headers

                            stream_info = loan_df[loan_df["Final Stream"] == stream]

                            if stream_info.empty:
                                st.error(f"❌ Fee and Budget details not found for stream: {stream}")
                            else:
                                fee = int(stream_info["Fee"].values[0])
                                budget = int(stream_info["Budget"].values[0])

                                # Ask for loan amount
                                loan_amount = None  # Initialize loan_amount
                                while loan_amount is None:  # Continue to ask until a valid amount is entered
                                    try:
                                        loan_amount = st.number_input(f"🎓 Your stream is '{stream}'. Fee: ₹{fee}. Enter loan amount (within ₹{budget} and ≥ ₹200000):", min_value=200000, max_value=budget)

                                        if loan_amount < 200000:
                                            st.warning("⚠️ Loan amount must be at least ₹200000.")
                                            loan_amount = None  # Reset to None to prompt the user again
                                        elif loan_amount > budget:
                                            st.warning(f"⚠️ Requested loan amount exceeds the budget (₹{budget}). Please enter a valid amount.")
                                            loan_amount = None  # Reset to None to prompt the user again
                                        else:
                                            break  # Valid input, exit the loop
                                    except ValueError:
                                        st.warning("⚠️ Please enter a numeric value.")
                                        loan_amount = None  # Reset to None to prompt the user again

                                due = fee - loan_amount
                                st.success(f"✅ Loan of ₹{loan_amount} granted to {student_name}.")
                                st.write(f"💰 Remaining due: ₹{due}")

                                # 8-Semester Distribution with 3% interest per semester
                                sem_distribution = []
                                base_installment = due / 8
                                for sem in range(1, 9):
                                    interest = base_installment * 0.03
                                    total = base_installment + interest
                                    sem_distribution.append([f"Semester {sem}", f"₹{base_installment:.2f}", f"₹{interest:.2f}", f"₹{total:.2f}"])

                                st.subheader("📊 Payment Schedule (8 Semesters @ 3% interest per semester):")
                                st.table(sem_distribution)
                        else:
                            st.error(f"❌ No data found for student: {student_name}")
            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.write("Enter a student name to check loan eligibility.")
        
        
if page == "Final":
    import streamlit as st
    import pandas as pd

    # ---- PAGE CONFIG ----
    # st.set_page_config(
    #     page_title="Student Records",
    #     layout="wide",
    #     page_icon="📄"
    # )

    # ---- TITLE ----
#    import streamlit as st

# ---- CUSTOM STYLING ----
    st.markdown("""
        <style>
            .small-title {
            font-size: 1.5rem;
            font-weight: 600;
            text-align: left;
            color: #2c3e50;
            }
        </style>
    """, unsafe_allow_html=True)

# ---- TITLE ----
    st.markdown('<div class="small-title">📄 Student Data Overview</div>', unsafe_allow_html=True)

    st.markdown("Here is the list of all registered students and their admission details:")

    # ---- READ CSV ----
    try:
        df = pd.read_csv("C:/Generative Ai/UniversityHelpdeskAutomation/Documents/students_with_eligibility.csv")  # Make sure the file path is correct
        st.dataframe(df, use_container_width=True)
    except FileNotFoundError:
        st.error("❌ CSV file not found. Please make sure 'students.csv' exists in the project directory.")
    except Exception as e:
        st.error(f"⚠️ An error occurred: {e}")


