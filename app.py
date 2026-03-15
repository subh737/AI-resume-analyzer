import streamlit as st
from PyPDF2 import PdfReader
import pickle
import re
import json
import sklearn 

import joblib
from huggingface_hub import hf_hub_download


st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)
# ==========================================================
# LOADING MODELS
# ==========================================================

@st.cache_resource
def load_models():

    REPO_ID = "Subh737/resume-analyzer-model"

    # Categorization Model
    rf_classifier_categorization_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="rf_classifier_categorization.pkl"
    )

    # Job Recommendation Model
    rf_classifier_job_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="rf_classifier_job_recommendation.pkl"
    )

    # Vectorizers
    tfidf_vectorizer_categorization_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="tfidf_vectorizer_categorization.pkl"
    )

    tfidf_vectorizer_job_path = hf_hub_download(
        repo_id=REPO_ID,
        filename="tfidf_vectorizer_job_recommendation.pkl"
    )

    # Load the models

    rf_classifier_categorization = joblib.load(rf_classifier_categorization_path)

    rf_classifier_job = joblib.load(rf_classifier_job_path)

    tfidf_vectorizer_categorization = joblib.load(tfidf_vectorizer_categorization_path)

    tfidf_vectorizer_job = joblib.load(tfidf_vectorizer_job_path)
    

    return (
    rf_classifier_categorization,
    rf_classifier_job,
    tfidf_vectorizer_categorization,
    tfidf_vectorizer_job
    )    

(
    rf_classifier_categorization,
    rf_classifier_job,
    tfidf_vectorizer_categorization,
    tfidf_vectorizer_job
) = load_models()

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_resume(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\S+", " ", text)
    text = re.sub(r"#\S+", " ", text)
    text = re.sub(r"[^\x00-\x7f]", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ==========================================================
# MODEL PREDICTIONS
# ==========================================================

def predict_category(text):
    text = clean_resume(text)
    vector = tfidf_vectorizer_categorization.transform([text])
    prediction = rf_classifier_categorization.predict(vector)
    return prediction[0]

def recommend_job(text):
    text = clean_resume(text)
    vector = tfidf_vectorizer_job.transform([text])
    prediction = rf_classifier_job.predict(vector)

    return prediction[0]
# ==========================================================
# PDF TO TEXT CONVERSION
# ==========================================================

def pdf_to_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text



# ==========================================================
# RESUME PARSER
# ==========================================================

# ==========================================================
# EMAIL EXTRACTION
# ==========================================================


def extract_email(text):
    # Finding all possible emails
    pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    emails = re.findall(pattern, text)

    if not emails:
        return None

    # Cleaning wrongly attached prefixes
    cleaned_emails = []
    for email in emails:
        # removing leading unwanted characters before actual username
        email = re.sub(r"^[^a-zA-Z0-9]+", "", email)

        # removing common merged prefixes like 'pe', 'ph', etc.
        email = re.sub(r"^(pe|ph|mo|mb|tel)+", "", email, flags=re.IGNORECASE)

        cleaned_emails.append(email)

    return cleaned_emails[0]


# ==========================================================
# PHONE NUMBER EXTRACTION
# ==========================================================

def extract_phone(text):

    contact_number = None

    # Regex pattern to detect phone numbers
    pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"

    match = re.search(pattern, text)

    if match:
        contact_number = match.group()

    return contact_number



def extract_name(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else None


# ================= SKILLS =================
SKILLS = [ 'Python', 'Data Analysis', 'Machine Learning', 'Communication', 'Project Management', 'Deep Learning', 'SQL',
        'Tableau',
        'Java', 'C++', 'JavaScript', 'HTML', 'CSS', 'React', 'Angular', 'Node.js', 'MongoDB', 'Express.js', 'Git',
        'Research', 'Statistics', 'Quantitative Analysis', 'Qualitative Analysis', 'SPSS', 'R', 'Data Visualization',
        'Matplotlib',
        'Seaborn', 'Plotly', 'Pandas', 'Numpy', 'Scikit-learn', 'TensorFlow', 'Keras', 'PyTorch', 'NLTK', 'Text Mining',
        'Natural Language Processing', 'Computer Vision', 'Image Processing', 'OCR', 'Speech Recognition',
        'Recommendation Systems',
        'Collaborative Filtering', 'Content-Based Filtering', 'Reinforcement Learning', 'Neural Networks',
        'Convolutional Neural Networks',
        'Recurrent Neural Networks', 'Generative Adversarial Networks', 'XGBoost', 'Random Forest', 'Decision Trees',
        'Support Vector Machines',
        'Linear Regression', 'Logistic Regression', 'K-Means Clustering', 'Hierarchical Clustering', 'DBSCAN',
        'Association Rule Learning',
        'Apache Hadoop', 'Apache Spark', 'MapReduce', 'Hive', 'HBase', 'Apache Kafka', 'Data Warehousing', 'ETL',
        'Big Data Analytics',
        'Cloud Computing', 'Amazon Web Services (AWS)', 'Microsoft Azure', 'Google Cloud Platform (GCP)', 'Docker',
        'Kubernetes', 'Linux',
        'Shell Scripting', 'Cybersecurity', 'Network Security', 'Penetration Testing', 'Firewalls', 'Encryption',
        'Malware Analysis',
        'Digital Forensics', 'CI/CD', 'DevOps', 'Agile Methodology', 'Scrum', 'Kanban', 'Continuous Integration',
        'Continuous Deployment',
        'Software Development', 'Web Development', 'Mobile Development', 'Backend Development', 'Frontend Development',
        'Full-Stack Development',
        'UI/UX Design', 'Responsive Design', 'Wireframing', 'Prototyping', 'User Testing', 'Adobe Creative Suite',
        'Photoshop', 'Illustrator',
        'InDesign', 'Figma', 'Sketch', 'Zeplin', 'InVision', 'Product Management', 'Market Research',
        'Customer Development', 'Lean Startup',
        'Business Development', 'Sales', 'Marketing', 'Content Marketing', 'Social Media Marketing', 'Email Marketing',
        'SEO', 'SEM', 'PPC',
        'Google Analytics', 'Facebook Ads', 'LinkedIn Ads', 'Lead Generation', 'Customer Relationship Management (CRM)',
        'Salesforce',
        'HubSpot', 'Zendesk', 'Intercom', 'Customer Support', 'Technical Support', 'Troubleshooting',
        'Ticketing Systems', 'ServiceNow',
        'ITIL', 'Quality Assurance', 'Manual Testing', 'Automated Testing', 'Selenium', 'JUnit', 'Load Testing',
        'Performance Testing',
        'Regression Testing', 'Black Box Testing', 'White Box Testing', 'API Testing', 'Mobile Testing',
        'Usability Testing', 'Accessibility Testing',
        'Cross-Browser Testing', 'Agile Testing', 'User Acceptance Testing', 'Software Documentation',
        'Technical Writing', 'Copywriting',
        'Editing', 'Proofreading', 'Content Management Systems (CMS)', 'WordPress', 'Joomla', 'Drupal', 'Magento',
        'Shopify', 'E-commerce',
        'Payment Gateways', 'Inventory Management', 'Supply Chain Management', 'Logistics', 'Procurement',
        'ERP Systems', 'SAP', 'Oracle',
        'Microsoft Dynamics', 'Tableau', 'Power BI', 'QlikView', 'Looker', 'Data Warehousing', 'ETL',
        'Data Engineering', 'Data Governance',
        'Data Quality', 'Master Data Management', 'Predictive Analytics', 'Prescriptive Analytics',
        'Descriptive Analytics', 'Business Intelligence',
        'Dashboarding', 'Reporting', 'Data Mining', 'Web Scraping', 'API Integration', 'RESTful APIs', 'GraphQL',
        'SOAP', 'Microservices',
        'Serverless Architecture', 'Lambda Functions', 'Event-Driven Architecture', 'Message Queues', 'GraphQL',
        'Socket.io', 'WebSockets'
                     'Ruby', 'Ruby on Rails', 'PHP', 'Symfony', 'Laravel', 'CakePHP', 'Zend Framework', 'ASP.NET', 'C#',
        'VB.NET', 'ASP.NET MVC', 'Entity Framework',
        'Spring', 'Hibernate', 'Struts', 'Kotlin', 'Swift', 'Objective-C', 'iOS Development', 'Android Development',
        'Flutter', 'React Native', 'Ionic',
        'Mobile UI/UX Design', 'Material Design', 'SwiftUI', 'RxJava', 'RxSwift', 'Django', 'Flask', 'FastAPI',
        'Falcon', 'Tornado', 'WebSockets',
        'GraphQL', 'RESTful Web Services', 'SOAP', 'Microservices Architecture', 'Serverless Computing', 'AWS Lambda',
        'Google Cloud Functions',
        'Azure Functions', 'Server Administration', 'System Administration', 'Network Administration',
        'Database Administration', 'MySQL', 'PostgreSQL',
        'SQLite', 'Microsoft SQL Server', 'Oracle Database', 'NoSQL', 'MongoDB', 'Cassandra', 'Redis', 'Elasticsearch',
        'Firebase', 'Google Analytics',
        'Google Tag Manager', 'Adobe Analytics', 'Marketing Automation', 'Customer Data Platforms', 'Segment',
        'Salesforce Marketing Cloud', 'HubSpot CRM',
        'Zapier', 'IFTTT', 'Workflow Automation', 'Robotic Process Automation (RPA)', 'UI Automation',
        'Natural Language Generation (NLG)',
        'Virtual Reality (VR)', 'Augmented Reality (AR)', 'Mixed Reality (MR)', 'Unity', 'Unreal Engine', '3D Modeling',
        'Animation', 'Motion Graphics',
        'Game Design', 'Game Development', 'Level Design', 'Unity3D', 'Unreal Engine 4', 'Blender', 'Maya',
        'Adobe After Effects', 'Adobe Premiere Pro',
        'Final Cut Pro', 'Video Editing', 'Audio Editing', 'Sound Design', 'Music Production', 'Digital Marketing',
        'Content Strategy', 'Conversion Rate Optimization (CRO)',
        'A/B Testing', 'Customer Experience (CX)', 'User Experience (UX)', 'User Interface (UI)', 'Persona Development',
        'User Journey Mapping', 'Information Architecture (IA)',
        'Wireframing', 'Prototyping', 'Usability Testing', 'Accessibility Compliance', 'Internationalization (I18n)',
        'Localization (L10n)', 'Voice User Interface (VUI)',
        'Chatbots', 'Natural Language Understanding (NLU)', 'Speech Synthesis', 'Emotion Detection',
        'Sentiment Analysis', 'Image Recognition', 'Object Detection',
        'Facial Recognition', 'Gesture Recognition', 'Document Recognition', 'Fraud Detection',
        'Cyber Threat Intelligence', 'Security Information and Event Management (SIEM)',
        'Vulnerability Assessment', 'Incident Response', 'Forensic Analysis', 'Security Operations Center (SOC)',
        'Identity and Access Management (IAM)', 'Single Sign-On (SSO)',
        'Multi-Factor Authentication (MFA)', 'Blockchain', 'Cryptocurrency', 'Decentralized Finance (DeFi)',
        'Smart Contracts', 'Web3', 'Non-Fungible Tokens (NFTs)']


def extract_skills(text):

    found_skills = []

    for skill in SKILLS:
        if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE):
            found_skills.append(skill)

    return found_skills


# ================= EDUCATION =================
EDUCATION = ['Computer Science', 'Information Technology', 'Software Engineering', 'Electrical Engineering', 'Mechanical Engineering', 'Civil Engineering',
        'Chemical Engineering', 'Biomedical Engineering', 'Aerospace Engineering', 'Nuclear Engineering', 'Industrial Engineering', 'Systems Engineering',
        'Environmental Engineering', 'Petroleum Engineering', 'Geological Engineering', 'Marine Engineering', 'Robotics Engineering', 'Biotechnology',
        'Biochemistry', 'Microbiology', 'Genetics', 'Molecular Biology', 'Bioinformatics', 'Neuroscience', 'Biophysics', 'Biostatistics', 'Pharmacology',
        'Physiology', 'Anatomy', 'Pathology', 'Immunology', 'Epidemiology', 'Public Health', 'Health Administration', 'Nursing', 'Medicine', 'Dentistry',
        'Pharmacy', 'Veterinary Medicine', 'Medical Technology', 'Radiography', 'Physical Therapy', 'Occupational Therapy', 'Speech Therapy', 'Nutrition',
        'Sports Science', 'Kinesiology', 'Exercise Physiology', 'Sports Medicine', 'Rehabilitation Science', 'Psychology', 'Counseling', 'Social Work',
        'Sociology', 'Anthropology', 'Criminal Justice', 'Political Science', 'International Relations', 'Economics', 'Finance', 'Accounting', 'Business Administration',
        'Management', 'Marketing', 'Entrepreneurship', 'Hospitality Management', 'Tourism Management', 'Supply Chain Management', 'Logistics Management',
        'Operations Management', 'Human Resource Management', 'Organizational Behavior', 'Project Management', 'Quality Management', 'Risk Management',
        'Strategic Management', 'Public Administration', 'Urban Planning', 'Architecture', 'Interior Design', 'Landscape Architecture', 'Fine Arts',
        'Visual Arts', 'Graphic Design', 'Fashion Design', 'Industrial Design', 'Product Design', 'Animation', 'Film Studies', 'Media Studies',
        'Communication Studies', 'Journalism', 'Broadcasting', 'Creative Writing', 'English Literature', 'Linguistics', 'Translation Studies',
        'Foreign Languages', 'Modern Languages', 'Classical Studies', 'History', 'Archaeology', 'Philosophy', 'Theology', 'Religious Studies',
        'Ethics', 'Education', 'Early Childhood Education', 'Elementary Education', 'Secondary Education', 'Special Education', 'Higher Education',
        'Adult Education', 'Distance Education', 'Online Education', 'Instructional Design', 'Curriculum Development'
        'Library Science', 'Information Science', 'Computer Engineering', 'Software Development', 'Cybersecurity', 'Information Security',
        'Network Engineering', 'Data Science', 'Data Analytics', 'Business Analytics', 'Operations Research', 'Decision Sciences',
        'Human-Computer Interaction', 'User Experience Design', 'User Interface Design', 'Digital Marketing', 'Content Strategy',
        'Brand Management', 'Public Relations', 'Corporate Communications', 'Media Production', 'Digital Media', 'Web Development',
        'Mobile App Development', 'Game Development', 'Virtual Reality', 'Augmented Reality', 'Blockchain Technology', 'Cryptocurrency',
        'Digital Forensics', 'Forensic Science', 'Criminalistics', 'Crime Scene Investigation', 'Emergency Management', 'Fire Science',
        'Environmental Science', 'Climate Science', 'Meteorology', 'Geography', 'Geomatics', 'Remote Sensing', 'Geoinformatics',
        'Cartography', 'GIS (Geographic Information Systems)', 'Environmental Management', 'Sustainability Studies', 'Renewable Energy',
        'Green Technology', 'Ecology', 'Conservation Biology', 'Wildlife Biology', 'Zoology']

def extract_education(text):
    
    education_section = ""

    section_pattern = r"(education[\s\S]*?)(experience|projects|skills|certifications|$)"
    match = re.search(section_pattern, text, re.IGNORECASE)

    if match:
        education_section = match.group(1)
    else:
        education_section = text  
    found_degrees = set()

   
    for keyword in EDUCATION:
        pattern = r"\b" + re.escape(keyword) + r"\b"
        if re.search(pattern, education_section, re.IGNORECASE):
            found_degrees.add(keyword)

    return list(found_degrees)    

# ==========================================================
# ATS SCORE CALCULATION
# ==========================================================
def calculate_ats_score(resume_text, job_description):

    if not job_description:
        return 0, [], []

    resume_words = set(re.findall(r'\w+', resume_text.lower()))
    jd_words = set(re.findall(r'\w+', job_description.lower()))

    # keyword matching
    matched_keywords = resume_words.intersection(jd_words)
    missing_keywords = jd_words.difference(resume_words)

    keyword_score = len(matched_keywords) / len(jd_words) * 100

  

    skills_found = [skill for skill in SKILLS if skill in resume_text.lower()]
    skills_score = (len(skills_found) / len(SKILLS)) * 100

    # resume length score
    word_count = len(resume_words)

    if 400 <= word_count <= 900:
        length_score = 100
    else:
        length_score = 60

    # section score
    sections = ["education", "experience", "skills", "projects"]
    section_found = sum(1 for sec in sections if sec in resume_text.lower())

    section_score = (section_found / len(sections)) * 100

    ats_score = (
        0.7 * keyword_score +
        0.1 * skills_score +
        0.1 * length_score +
        0.1 * section_score
    )

    return round(ats_score, 2), list(matched_keywords)[:20], list(missing_keywords)[:20]


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home", "Resume Analyzer", "About"]
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "Home":

    st.title("🤖 AI Resume Screening System")

    st.markdown("""
    ### Features

    ✔ Resume Categorization using Machine Learning  
    ✔ AI Job Recommendation  
    ✔ Resume Information Extraction  
    ✔ Skill Detection  
    ✔ Education Detection  

    Upload a resume to get instant AI insights.
    """)


# ==========================================================
# RESUME ANALYZER
# ==========================================================

elif page == "Resume Analyzer":

    st.title("📄 Resume Analyzer")
 

    uploaded_file = st.file_uploader(
        "Upload Resume (PDF or TXT)",
        type=["pdf", "txt"]
    )
    st.subheader("📄 Job Description")

    job_description = st.text_area(
    "Paste Job Description here to calculate ATS score",
    height=200
    )

    if uploaded_file:

        with st.spinner("Analyzing Resume..."):

            if uploaded_file.type == "application/pdf":
                text = pdf_to_text(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")

            # ML predictions
            category = predict_category(text)
            job = recommend_job(text)

            st.subheader("🎯 ATS Resume Score")

            ats_score, matched_keywords, missing_keywords = calculate_ats_score(
            text, job_description
            )

            st.metric("ATS Score", f"{ats_score}%")

            st.progress(int(ats_score))

            col1, col2 = st.columns(2)

            with col1:
              st.success("✅ Matched Keywords")
              st.write(", ".join(matched_keywords))

            with col2:
              st.error("❌ Missing Keywords")
              st.write(", ".join(missing_keywords))

            # Resume parsing
            name = extract_name(text)
            email = extract_email(text)
            phone = extract_phone(text)
            skills = extract_skills(text)
            education = extract_education(text)

            st.success("Resume analyzed successfully!")

            

                  
        
        # ==================================================
        # PREDICTIONS
        # ==================================================

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Predicted Category", category)

        with col2:
            st.metric("Recommended Job", job)

        # ==================================================
        # PERSONAL DETAILS
        # ==================================================

        st.subheader("📌 Personal Information")

        c1, c2, c3 = st.columns(3)

        c1.info(f"👤 Name: {name}")
        c2.info(f"📧 Email: {email}")
        c3.info(f"📱 Phone: {phone}")

        # ==================================================
        # SKILLS
        # ==================================================

        st.subheader("🧠 Detected Skills")

        if skills:
            st.write(" | ".join(skills))
        else:
            st.write("No skills detected")

        # ==================================================
        # EDUCATION
        # ==================================================

        st.subheader("🎓 Education")

        if education:
            st.write(" | ".join(education))
        else:
            st.write("No education keywords found")

        # ==================================================
        # DOWNLOAD PARSED DATA
        # ==================================================

        result = {
            "name": name,
            "email": email,
            "phone": phone,
            "skills": skills,
            "education": education,
            "category": category,
            "recommended_job": job
        }

        st.download_button(
            "⬇ Download Parsed Resume JSON",
            json.dumps(result, indent=4),
            file_name="parsed_resume.json"
        )
      
# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "About":

    st.title("ℹ About This Project")

    st.markdown("""
    **AI Resume Screening System**

    This project uses Machine Learning to:

    - Categorize resumes
    - Recommend suitable job roles
    - Extract important information from resumes

    Technologies Used:

    - Python
    - Streamlit
    - Scikit-learn
    - NLP
    - PDF Parsing
    """)