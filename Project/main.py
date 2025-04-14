import streamlit as st

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="NovaTech Admission Portal",
    layout="centered",
    page_icon="🎓"
)

# ---- MINIMAL STYLING ----
st.markdown("""
    <style>
        .title {
            font-size: 2rem;
            font-weight: 600;
            text-align: center;
            color: #1a237e;
            margin-bottom: 0.5rem;
        }
        .tagline {
            text-align: center;
            color: #455a64;
            margin-bottom: 2rem;
        }
        .section {
            margin-bottom: 2rem;
        }
        .section-title {
            font-size: 1.2rem;
            color: #1a237e;
            margin-bottom: 0.5rem;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 0.3rem;
        }
    </style>
""", unsafe_allow_html=True)

# ---- HEADER ----
st.markdown('<div class="title">NovaTech University</div>', unsafe_allow_html=True)
st.markdown('<div class="tagline">Admission Portal</div>', unsafe_allow_html=True)

# ---- ADMISSION INFO ----
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Admission Process</div>', unsafe_allow_html=True)
st.markdown("""
1. **Explore Programs**: Browse our academic offerings
2. **Review Requirements**: Check admission criteria
3. **Submit Documents**: Prepare required materials
4. **Application Review**: Our team will evaluate your submission
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---- CONTACT ----
st.markdown('<div class="section">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Contact Information</div>', unsafe_allow_html=True)
st.markdown("""
📧 **Email**: admissions@novatech.edu  
📞 **Phone**: (555) 123-4567  
🏢 **Office Hours**: Monday-Friday, 9AM-5PM
""")
st.markdown('</div>', unsafe_allow_html=True)

# ---- FOOTER ----
st.markdown("---")
st.caption("© 2025 NovaTech University")