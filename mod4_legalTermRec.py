import re

# -----------------------------------------------------------
# Custom Legal Term Dictionary
# -----------------------------------------------------------
legal_terms = {
    "indemnity": "Security or protection against a loss or other financial burden.",
    "arbitration": "A method of resolving disputes outside the courts.",
    "force majeure": "Unforeseeable circumstances that prevent someone from fulfilling a contract.",
    "breach": "A violation of a law, duty, or other form of obligation.",
    "jurisdiction": "The official power to make legal decisions and judgments.",
    "confidentiality": "The obligation to keep certain information private or secret.",
    "termination": "The act of ending a contract or agreement.",
    "governing law": "The legal jurisdiction whose laws will apply to the agreement.",
    "liability": "Legal responsibility for one’s acts or omissions.",
    "warranty": "A promise or assurance regarding the condition or performance of something."
}

# -----------------------------------------------------------
# Function: Recognize Legal Terms in Text
# -----------------------------------------------------------
def recognize_legal_terms(text: str, term_dict: dict):
    """
    Identify legal terms present in the given text based on the custom dictionary.
    Returns a dictionary of recognized terms and their definitions.
    """
    found_terms = {}
    for term in term_dict:
        if re.search(r'\b' + re.escape(term) + r'\b', text.lower()):
            found_terms[term] = term_dict[term]
    return found_terms


# -----------------------------------------------------------
# Example Usage
# -----------------------------------------------------------
if __name__ == "__main__":
    contract_text = """
    This Agreement shall be governed by the laws of India. 
    Any dispute arising under this Agreement shall be resolved by arbitration. 
    The Company shall not be liable for breach of confidentiality or warranty.
    """

    recognized = recognize_legal_terms(contract_text, legal_terms)

    print("✅ Recognized Legal Terms:\n")
    if recognized:
        for term, definition in recognized.items():
            print(f"🔹 {term}: {definition}")
    else:
        print("No legal terms recognized.")
