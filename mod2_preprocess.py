#mod2

import re
import nltk
import spacy
from nltk.tokenize import sent_tokenize
from mod1_docingestion import extract_text

# ---------------- Initialization ----------------
nltk.download('punkt', quiet=True)
nlp = spacy.load("en_core_web_sm")


# ---------------- TEXT CLEANING ----------------
def clean_text(text: str) -> str:
    """
    Normalize and clean legal text.
    - Remove unwanted spaces, tabs, and special characters
    - Standardize quotes
    """
    if not text:
        return ""

    text = text.replace('\xa0', ' ')  # non-breaking space
    text = re.sub(r'[\t\r\f\v]', ' ', text)  # control chars
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    text = re.sub(r'[“”]', '"', text)  # curly to straight quotes
    text = re.sub(r"[’‘]", "'", text)  # curly apostrophes to straight
    return text.strip()


# ---------------- CLAUSE SEGMENTATION ----------------
def segment_clauses(text: str) -> list:
    """
    Segment text into clauses based on numbering patterns.
    Supports: 1.1, 2.3(a), 3.1.2, etc.
    """
    if not text:
        return []

    # Regex: detects legal numbering (e.g., 1.1, 2.3(a), 3.1.2)
    pattern = re.compile(r'(?=\n?\d{1,2}(\.\d{1,2})+[\)\.]?\s)')
    parts = pattern.split(text)
    clauses, temp = [], ""

    for part in parts:
        if re.match(r'\d{1,2}(\.\d{1,2})+[\)\.]?', part.strip()):
            if temp:
                clauses.append(temp.strip())
            temp = part
        else:
            temp += part
    if temp:
        clauses.append(temp.strip())

    return clauses if clauses else [text.strip()]


# ---------------- SENTENCE SPLITTING ----------------
def split_sentences(text: str) -> list:
    """
    Split a clause into individual sentences using NLTK.
    """
    if not text:
        return []
    return sent_tokenize(text)


# ---------------- NAMED ENTITY EXTRACTION ----------------
def extract_entities(text: str) -> list:
    """
    Extract named entities (ORG, DATE, MONEY, etc.) using spaCy.
    Returns a list of (entity, label).
    """
    if not text:
        return []
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

def preprocess_clause(clause_text: str) -> dict:
    """
    Process a single clause:
    - Clean text
    - Split into sentences
    - Extract entities
    """
    cleaned = clean_text(clause_text)
    sentences = split_sentences(cleaned)
    entities = extract_entities(cleaned)

    return {
        "raw_text": clause_text,
        "cleaned_text": cleaned,
        "sentences": sentences,
        "entities": entities
    }

# ----------- BATCH PROCESSING -----------

def preprocess_contract_text(raw_text: str) -> list:
    """
    Preprocess an entire contract's text:
    - Clean
    - Segment into clauses
    - Process each clause
    """
    cleaned_text = clean_text(raw_text)
    clauses = segment_clauses(cleaned_text)

    processed = []
    for clause in clauses:
        result = preprocess_clause(clause)
        processed.append(result)

    return processed






# Import extract_text from doc1.py

# import sys
# sys.path.append(r"d:\Internships\Infosys SpringBoard\ClauseEase")



# Specify your contract file path (PDF or DOCX)
contract_file = r"d:\Internships\Infosys SpringBoard\ClauseEase\employment_agreement.pdf"
contract_text = extract_text(contract_file)

processed = preprocess_contract_text(contract_text)

# Example: print the first clause's data
print(processed[0])