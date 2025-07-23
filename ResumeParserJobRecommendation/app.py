import streamlit as st
import os
import pandas as pd
import re

from parser import parse_resume
from recommendation import build_tfidf_matrix, recommend_jobs

def main():
    st.title("Resume Parser and Job Recommendation System")

    uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=["pdf", "docx"])
    if uploaded_file is not None:
        # Save uploaded file temporarily
        _, ext = os.path.splitext(uploaded_file.name)
        temp_file_path = f"temp_resume{ext}"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Load skills list - for demo, a static list
        skills_list = ["Python", "Java", "Machine Learning", "NLP", "Data Analysis", "SQL", "Deep Learning"]

        # Parse resume
        parsed_data = parse_resume(temp_file_path, skills_list)
        st.subheader("Parsed Resume Data")
        st.json(parsed_data)

    st.write(f"Current working directory: {os.getcwd()}")
    dataset_path = "reduced_postings.csv"
    st.write(f"Checking if dataset file exists at: {dataset_path} -> {os.path.exists(dataset_path)}")

    job_df = pd.read_csv(dataset_path)

    st.write(f"Job dataset columns: {list(job_df.columns)}")  # Debug print columns
    st.write("Job dataset sample rows:")
    st.write(job_df.head(10))  # Debug print first 10 rows

    # Normalize column names by stripping whitespace and lowercasing
    job_df.columns = job_df.columns.str.strip().str.lower()

    # Rename columns to expected names with normalized keys
    rename_map = {
        'title': 'job_title',
        'skills_desc': 'cleaned_description',
        'application_url': 'application_link',
        'application_link': 'application_link'  # in case already named so
    }
    job_df = job_df.rename(columns={k: v for k, v in rename_map.items() if k in job_df.columns})

    # Fill missing cleaned_description with description column if needed
    if 'cleaned_description' not in job_df or job_df['cleaned_description'].isnull().all():
        if 'description' in job_df.columns:
            job_df['cleaned_description'] = job_df['description'].fillna('')
        else:
            job_df['cleaned_description'] = ''

    # Preprocess cleaned_description: lowercase and remove special chars
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', str(text))
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    job_df['cleaned_description'] = job_df['cleaned_description'].apply(clean_text)

    tfidf_matrix, vectorizer = build_tfidf_matrix(job_df['cleaned_description'])

    # Recommend jobs
    recommendations = recommend_jobs(parsed_data, job_df, tfidf_matrix, vectorizer, top_n=5)
    st.subheader("Job Recommendations")
    for idx, row in recommendations.iterrows():
        if 'application_link' in row and pd.notna(row['application_link']) and row['application_link'].strip() != '':
            st.markdown(f"[**{row['job_title']}**]({row['application_link']}) - Similarity Score: {row['similarity_score']:.2f}")
            st.markdown(f"Apply here: [Link]({row['application_link']})")
        else:
            st.markdown(f"**{row['job_title']}** - Similarity Score: {row['similarity_score']:.2f}")

if __name__ == "__main__":
    main()
