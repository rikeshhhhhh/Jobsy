import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def build_tfidf_matrix(job_descriptions):
    """
    Build TF-IDF matrix from job descriptions.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(job_descriptions)
    return tfidf_matrix, vectorizer

def recommend_jobs(parsed_resume_data, job_dataset, tfidf_matrix, vectorizer, top_n=5):
    """
    Recommend jobs based on parsed resume data and job descriptions similarity.
    - parsed_resume_data: dict with keys like 'skills', 'education', 'work_experience', 'projects_certifications'
    - job_dataset: pandas DataFrame with job data including 'cleaned_description'
    - tfidf_matrix: TF-IDF matrix of job descriptions
    - vectorizer: TF-IDF vectorizer used to transform text
    - top_n: number of top recommendations to return
    """
    # Create a query string from multiple resume fields
    query_parts = []
    if 'skills' in parsed_resume_data and parsed_resume_data['skills']:
        query_parts.append(" ".join(parsed_resume_data['skills']))
    if 'education' in parsed_resume_data and parsed_resume_data['education']:
        query_parts.append(" ".join(parsed_resume_data['education']))
    if 'work_experience' in parsed_resume_data and parsed_resume_data['work_experience']:
        query_parts.append(" ".join(parsed_resume_data['work_experience']))
    if 'projects_certifications' in parsed_resume_data and parsed_resume_data['projects_certifications']:
        query_parts.append(" ".join(parsed_resume_data['projects_certifications']))
    query = " ".join(query_parts).lower()

    # Transform query to TF-IDF vector
    query_vec = vectorizer.transform([query])

    # Compute cosine similarity between query and job descriptions
    cosine_sim = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Boost similarity score for priority roles in job title
    priority_roles = [
        'data scientist', 'data analyst', 'machine learning engineer', 'ml engineer', 'data engineer', 'ai engineer',
        'developer', 'engineer', 'analyst', 'manager', 'consultant', 'business analyst', 'salesforce developer',
        'principal salesforce developer', 'engineering manager', 'software design quality'
    ]
    boost_factor = 0.2  # Boost amount to add

    boosted_sim = []
    for idx, row in job_dataset.iterrows():
        score = cosine_sim[idx]
        title = str(row.get('job_title', '')).lower()
        if any(role in title for role in priority_roles):
            score += boost_factor
        boosted_sim.append(score)

    boosted_sim = pd.Series(boosted_sim)

    # Get top N job indices based on boosted similarity
    top_indices = boosted_sim.argsort()[-top_n:][::-1]

    # Return top N job recommendations with boosted similarity scores
    recommendations = job_dataset.iloc[top_indices].copy()
    recommendations['similarity_score'] = boosted_sim[top_indices].values
    return recommendations
