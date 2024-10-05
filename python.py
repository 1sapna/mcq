import spacy
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
    nlp = spacy.load("en_core_web_sm")
  except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    try:
      subprocess.check_call(["spacy", "download", "en_core_web_sm"])
    except subprocess.CalledProcessError as e:
      st.write("Error downloading spaCy model. Please check your internet connection.")
      st.write(str(e))
      st.stop()
  return nlp

nlp = ensure_spacy_model()

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
