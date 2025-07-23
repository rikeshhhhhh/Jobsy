import re
from pdfminer.high_level import extract_text
from docx import Document
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_path):
    """Extract full text from a PDF file."""
    return extract_text(pdf_path)

def extract_text_from_docx(docx_path):
    """Extract full text from a DOCX file."""
    doc = Document(docx_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def classify_section(line):
    """Classify the section of a resume line using keyword matching and NLP."""
    line_lower = line.lower()
    if any(keyword in line_lower for keyword in ['education', 'bachelor', 'master', 'phd', 'university', 'college', 'school']):
        return 'education'
    elif any(keyword in line_lower for keyword in ['experience', 'work experience', 'employment', 'professional experience', 'career']):
        return 'work_experience'
    elif any(keyword in line_lower for keyword in ['project', 'certification', 'training', 'course']):
        return 'projects_certifications'
    elif any(keyword in line_lower for keyword in ['skill', 'technical skills', 'skills']):
        return 'skills'
    else:
        return 'other'

def parse_resume(file_path, skills_list):
    """
    Parse resume file and extract structured information.
    Returns a dictionary with keys: name, email, phone, skills, education, work_experience, projects_certifications.
    """
    if file_path.lower().endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.lower().endswith('.docx'):
        text = extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Only PDF and DOCX are supported.")

    lines = text.split('\n')
    name = None
    for line in lines:
        stripped = line.strip()
        if stripped and re.match(r'^[A-Za-z\s]+$', stripped) and len(stripped) > 2:
            name = stripped
            break

    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    email = email_match.group(0) if email_match else None

    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{2,4}\)?[-.\s]?)?[\d\s.-]{6,}', text)
    phone = phone_match.group(0).strip() if phone_match else None

    # Initialize sections
    education = []
    work_experience = []
    projects_certifications = []
    skills_section = []

    current_section = 'other'
    buffer = []
    for line in lines:
        section = classify_section(line)
        if section != 'other':
            if section == current_section:
                # Continue current section, add previous buffer lines
                buffer.append(line.strip())
                continue
            # Save previous buffer to the previous section
            if current_section == 'education':
                education.extend([l for l in buffer if l.strip() != ''])
            elif current_section == 'work_experience':
                work_experience.extend([l for l in buffer if l.strip() != ''])
            elif current_section == 'projects_certifications':
                projects_certifications.extend([l for l in buffer if l.strip() != ''])
            elif current_section == 'skills':
                skills_section.extend([l for l in buffer if l.strip() != ''])
            buffer = [line.strip()]
            current_section = section
            continue
        buffer.append(line.strip())
    # Add remaining buffer
    if current_section == 'education':
        education.extend([l for l in buffer if l.strip() != ''])
    elif current_section == 'work_experience':
        work_experience.extend([l for l in buffer if l.strip() != ''])
    elif current_section == 'projects_certifications':
        projects_certifications.extend([l for l in buffer if l.strip() != ''])
    elif current_section == 'skills':
        skills_section.extend([l for l in buffer if l.strip() != ''])

    # Extract skills by matching keywords from skills_list and skills_section text
    skills_found = []
    text_lower = text.lower()
    for skill in skills_list:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            skills_found.append(skill)
    # Also add skills from skills_section lines
    for line in skills_section:
        for skill in skills_list:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, line.lower()):
                if skill not in skills_found:
                    skills_found.append(skill)

    # Extract keywords from work experience and projects/certifications for better recommendation
    def extract_keywords(text_lines):
        keywords = []
        for line in text_lines:
            words = re.findall(r'\b\w{3,}\b', line.lower())
            keywords.extend(words)
        return list(set(keywords))

    work_exp_keywords = extract_keywords(work_experience)
    proj_cert_keywords = extract_keywords(projects_certifications)

    # Clean and preprocess education and work experience text to extract meaningful keywords
    def preprocess_text(text_lines):
        cleaned = []
        for line in text_lines:
            # Remove special characters and digits, keep words only
            line_clean = re.sub(r'[^a-zA-Z\s]', ' ', line)
            words = [w for w in line_clean.lower().split() if len(w) > 2]
            cleaned.extend(words)
        return list(set(cleaned))

    education_keywords = preprocess_text(education)
    work_experience_keywords = preprocess_text(work_experience)

    combined_skills = list(set(skills_found + work_exp_keywords + proj_cert_keywords + education_keywords + work_experience_keywords))

    parsed_data = {
        "name": name,
        "email": email,
        "phone": phone,
        "skills": combined_skills,
        "education": education,
        "work_experience": work_experience,
        "projects_certifications": projects_certifications
    }

    return parsed_data
