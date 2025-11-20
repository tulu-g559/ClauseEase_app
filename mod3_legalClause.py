# fast_clause_classifier_bertmini.py
# -----------------------------------------------------------
# Fast clause classification using prajjwal1/bert-mini (CPU-friendly)
# -----------------------------------------------------------

from transformers import AutoTokenizer, AutoModel
import torch
from mod1_docingestion import extract_text
from mod2_preprocess import preprocess_contract_text

# -----------------------------------------------------------
# Load Lightweight BERT-Mini model
# -----------------------------------------------------------
MODEL_NAME = "prajjwal1/bert-mini"  # small + fast (22M params)
device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

# -----------------------------------------------------------
# Clause Categories with example phrases (semantic anchors)
# -----------------------------------------------------------
clause_labels = {
    "Confidentiality": [
        "keep information confidential",
        "non-disclosure agreement",
        "maintain secrecy of company information",
        "privacy and confidentiality"
    ],
    "Termination": [
        "termination of agreement",
        "end of employment",
        "cancellation of contract",
        "termination notice period"
    ],
    "Indemnity": [
        "indemnify and hold harmless",
        "compensation for damages",
        "liability clause",
        "responsible for any losses"
    ],
    "Dispute Resolution": [
        "arbitration",
        "dispute resolution procedure",
        "settlement of disputes",
        "mediation or arbitration clause"
    ],
    "Governing Law": [
        "governed by laws",
        "jurisdiction",
        "governing law of india",
        "applicable law clause"
    ]
}

# -----------------------------------------------------------
# Helper functions
# -----------------------------------------------------------
def mean_pooling(model_output, attention_mask):
    """Compute mean pooling over token embeddings."""
    token_embeddings = model_output[0]
    mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)


def encode_texts(texts):
    """Encode a list of texts into normalized embeddings."""
    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors='pt', max_length=128
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs, inputs['attention_mask'])
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


# Precompute embeddings for all label anchors
label_texts = [phrase for label_list in clause_labels.values() for phrase in label_list]
label_names = [k for k, v in clause_labels.items() for _ in v]
label_embeds = encode_texts(label_texts)


def detect_clause_type(text: str):
    """Predict the most relevant clause type using cosine similarity."""
    if not text.strip():
        return "Unknown", 0.0

    text_embed = encode_texts([text])
    cosine_sim = torch.nn.functional.cosine_similarity(text_embed, label_embeds)
    best_idx = torch.argmax(cosine_sim).item()
    best_label = label_names[best_idx]
    confidence = float(cosine_sim[best_idx])
    return best_label, confidence


# -----------------------------------------------------------
# Example Usage (Integrated with your pipeline)
# -----------------------------------------------------------
if __name__ == "__main__":
    contract_file = r"d:\Internships\Infosys SpringBoard\ClauseEase\employment_agreement.pdf"
    contract_text = extract_text(contract_file)

    if contract_text.startswith("[ERROR]"):
        print(contract_text)
    else:
        processed_clauses = preprocess_contract_text(contract_text)
        print("✅ Running Fast Clause Detection with BERT-Mini...\n")

        for i, clause in enumerate(processed_clauses[:10]):  # limit to first 10 clauses
            clause_text = clause["cleaned_text"]
            label, confidence = detect_clause_type(clause_text)
            print(f"Clause {i+1}: {label} (confidence: {confidence:.2f})")
            print(clause_text)
            print("-" * 80)
