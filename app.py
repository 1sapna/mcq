import spacy
import subprocess
import sys
import pdfplumber
import pytesseract
from PIL import Image
import streamlit as st
from collections import Counter
import random
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to ensure spaCy model is installed
def ensure_spacy_model():
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        st.write("Downloading spaCy model 'en_core_web_sm'...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

ensure_spacy_model()

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

### Text Extraction Functions for PDF, JPG, and Direct Text

# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Extract text from a JPG or image file using OCR
def extract_text_from_image(image_path):
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

### Utility Functions for Text Cleaning and Synonyms

# Clean the extracted text (remove unwanted characters)
def clean_text(text):
    text = re.sub(r'\n+', ' ', text)  # Replace multiple newlines with a space
    text = re.sub(r'[^\w\s.,;]', '', text)  # Keep words, spaces, and basic punctuation
    return text.strip()

# Get synonyms of a word using WordNet (NLTK)
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.add(lemma.name())
    return list(synonyms)

# Get the top N most important words using TF-IDF (as potential distractors)
def get_top_n_words(text, n=10):
    tfidf = TfidfVectorizer(max_features=n)
    response = tfidf.fit_transform([text])
    feature_names = tfidf.get_feature_names_out()
    return feature_names

### MCQ Generation Function

def generate_mcqs(text, num_questions=5, difficulty='medium'):
    # Process the text with spaCy to split into sentences
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Select random sentences to generate questions
    selected_sentences = random.sample(sentences, min(num_questions, len(sentences)))

    # Initialize list to store generated MCQs
    mcqs = []
    
    # Generate MCQs for each selected sentence
    for sentence in selected_sentences:
        sent_doc = nlp(sentence)
        
        # Extract nouns from the sentence
        nouns = [token.text for token in sent_doc if token.pos_ == "NOUN"]
        
        # Skip if not enough nouns to form questions
        if len(nouns) < 2:
            continue

        # Count the occurrence of each noun
        noun_counts = Counter(nouns)
        if noun_counts:
            # Choose the most common noun as the subject (correct answer)
            subject = noun_counts.most_common(1)[0][0]

            # Replace the subject in the sentence with a blank to form the question
            question_stem = sentence.replace(subject, "__________")
            answer_choices = [subject]

            # Generate distractors based on difficulty level
            if difficulty == 'easy':
                distractors = random.sample(nouns, 2) + ['simple_word']
            elif difficulty == 'medium':
                distractors = random.sample(nouns, 3)
            else:  # 'hard' difficulty
                distractors = get_synonyms(subject)[:3]
            
            # Add distractors and shuffle the answer choices
            answer_choices.extend(distractors)
            random.shuffle(answer_choices)

            # Determine the correct answer's position (convert index to letter)
            correct_answer = chr(64 + answer_choices.index(subject) + 1)  # A, B, C, ...

            # Append the generated MCQ (question stem, answer choices, correct answer)
            mcqs.append((question_stem, answer_choices, correct_answer))

    return mcqs

### Streamlit Interface

# Title of the Streamlit app
st.title("MCQ Generator from Text, PDF, or Image")

# File uploader for PDF and JPG input
uploaded_file = st.file_uploader("Upload a PDF, JPG Image or Enter Text", type=["pdf", "jpg", "jpeg"])

# Text area for direct text input
text_input = st.text_area("Or Enter Text Manually")

# Number of MCQs to generate
num_questions = st.slider("Number of MCQs to generate", min_value=1, max_value=10, value=5)

# Difficulty level
difficulty = st.selectbox("Select difficulty level", ["easy", "medium", "hard"])

# Process and generate MCQs when the "Generate MCQs" button is clicked
if st.button("Generate MCQs"):
    text = ""
    
    # Process uploaded file (PDF or JPG)
    if uploaded_file:
        file_type = uploaded_file.type
        if "pdf" in file_type:
            text = extract_text_from_pdf(uploaded_file)
        elif "jpg" in file_type or "jpeg" in file_type:
            text = extract_text_from_image(uploaded_file)
    
    # Use the entered text
    elif text_input:
        text = text_input

    # Clean the text
    cleaned_text = clean_text(text)
    
    # Generate MCQs
    if cleaned_text:
        mcqs = generate_mcqs(cleaned_text, num_questions=num_questions, difficulty=difficulty)
        
        # Display the generated MCQs
        if mcqs:
            for i, (question, choices, answer) in enumerate(mcqs, 1):
                st.write(f"**Q{i}: {question}**")
                for j, choice in enumerate(choices, 1):
                    st.write(f"   {chr(64 + j)}. {choice}")
                st.write(f"**Answer: {answer}**\n")
        else:
            st.write("Not enough nouns or text to generate MCQs.")
    else:
        st.write("No valid text found to generate MCQs.")
