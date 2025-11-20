# fast_simplifier_ultra.py
from transformers import pipeline, AutoTokenizer
import nltk
nltk.download('punkt', quiet=True)
from nltk.tokenize import sent_tokenize
import time
import re
from difflib import SequenceMatcher

# -----------------------------------------------------
# Lightweight rule-based mapping (unchanged)
# -----------------------------------------------------
LEGAL_EASE_MAP = {
    'Notwithstanding anything to the contrary contained herein,': 'Regardless of other statements in this document,',
    'Notwithstanding anything to the contrary': 'Regardless of other statements',
    'hereinafter referred to as': 'called',
    "including but not limited to": "including",
    'shall': 'must',
    'herein': 'in this document',
    'pursuant to': 'under',
    'in the event of': 'if',
    'for the avoidance of doubt': 'to be clear',
}

def _apply_basic_rules(sentence: str) -> str:
    s = sentence
    for k, v in LEGAL_EASE_MAP.items():
        s = s.replace(k, v)
    return ' '.join(s.split())


# -----------------------------------------------------
# Use an ultra-small paraphraser
# -----------------------------------------------------'

MODEL_NAME = "google/flan-t5-small"

simplifier = pipeline(
    "text2text-generation",
    model=MODEL_NAME,
    device=-1,   # CPU only
)

# Initialize tokenizer for token counting
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def _count_tokens(text: str) -> int:
    """Count tokens in text using the model's tokenizer."""
    try:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        return len(tokens)
    except Exception:
        # Fallback: approximate by word count * 1.3 (average tokens per word)
        return int(len(text.split()) * 1.3)


def _group_sentences_into_paragraphs(sentences: list, sentences_per_para: int = 2) -> list:
    """
    Group sentences into short paragraphs for better context.
    Returns list of paragraph strings.
    """
    if not sentences:
        return []
    
    paragraphs = []
    for i in range(0, len(sentences), sentences_per_para):
        para_sentences = sentences[i:i + sentences_per_para]
        paragraph = ' '.join(para_sentences)
        paragraphs.append(paragraph)
    
    return paragraphs


def _split_paragraph_into_sentences(paragraph: str) -> list:
    """Split a paragraph back into individual sentences."""
    return [s.strip() for s in sent_tokenize(paragraph) if s.strip()]


def _similarity_score(text1: str, text2: str) -> float:
    """Calculate similarity between two texts (0.0 to 1.0)."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def _remove_repetitive_phrases(text: str) -> str:
    """Remove common repetitive legal phrases."""
    repetitive_patterns = [
        r'\b(?:the\s+)?(?:party|parties|lessee|lessor|tenant|landlord)\s+(?:hereby|shall|agrees?|acknowledges?)\s+(?:that|to)\b',
        r'\b(?:it\s+is\s+)?(?:hereby|expressly)\s+(?:agreed|understood|acknowledged)\s+(?:that|by)\b',
        r'\b(?:for\s+the\s+)?(?:avoidance\s+of\s+doubt|purposes\s+of\s+clarity)\b',
    ]
    for pattern in repetitive_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    # Clean up extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _remove_duplicate_sentences(sentences: list, similarity_threshold: float = 0.85) -> list:
    """
    Remove duplicate or near-duplicate sentences.
    similarity_threshold: 0.85 means 85% similar sentences are considered duplicates.
    """
    if not sentences:
        return []
    
    unique_sentences = []
    for sent in sentences:
        sent_clean = sent.strip()
        if not sent_clean:
            continue
        
        # Check if this sentence is too similar to any already added
        is_duplicate = False
        for existing in unique_sentences:
            similarity = _similarity_score(sent_clean, existing)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_sentences.append(sent_clean)
    
    return unique_sentences


# -----------------------------------------------------
# Batch paraphrase with improved prompt and dynamic length
# -----------------------------------------------------
def _paraphrase_batch(texts, num_beams=2, batch_size=12, prompt_template="simplify and shorten: {text}"):
    """
    Paraphrase a batch of texts with dynamic max_length based on input token count.
    Uses 'simplify and shorten' prompt for better results.
    """
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            # Calculate max_length for each text (60% of input tokens)
            prompts = []
            max_lengths = []
            for text in batch:
                input_tokens = _count_tokens(text)
                # Set max_length to 60% of input, but ensure minimum of 20 and maximum of 100
                calculated_max = max(20, min(100, int(input_tokens * 0.6)))
                max_lengths.append(calculated_max)
                prompts.append(prompt_template.format(text=text))
            
            # Use the average max_length for the batch (models expect single value)
            avg_max_length = int(sum(max_lengths) / len(max_lengths)) if max_lengths else 60
            
            outputs = simplifier(
                prompts,
                max_length=avg_max_length,
                num_beams=num_beams,
                early_stopping=True
            )
            
            for o in outputs:
                text = o.get('generated_text', '').strip()
                if not text:
                    # Fallback: use rule-based if model returns empty
                    idx = len(results)
                    if idx < len(batch):
                        text = _apply_basic_rules(batch[idx])
                results.append(text)
        except Exception as e:
            # Fallback: rule-based simplification
            for s in batch:
                results.append(_apply_basic_rules(s))
    return results


def _normalize_sentence(sentence: str) -> str:
    s = sentence.strip()
    if not s:
        return ""
    if s[0].islower():
        s = s[0].upper() + s[1:]
    if s[-1] not in '.!?':
        s = f"{s}."
    return s


# -----------------------------------------------------
# Simplify text with improved processing
# -----------------------------------------------------
def simplify_text(text, level='intermediate', max_length=60, fast=True):
    """
    Simplify text at one of three levels:
      - 'basic'       : rule-based only (no AI model)
      - 'intermediate': one-pass AI paraphrase with paragraph grouping
      - 'advanced'    : double-pass AI paraphrase with paragraph grouping
    
    Improvements:
      - Groups sentences into paragraphs (2-3 sentences) for better context
      - Uses "simplify and shorten" prompt for better results
      - Sets max_length to 60% of input token count to force shorter output
      - Post-processes to remove duplicate sentences and repetitive phrases
    """
    level = (level or 'intermediate').lower()
    sentences = [s.strip() for s in sent_tokenize(text or '') if s.strip()]

    if not sentences:
        return ""

    # Basic level: rule-based only, no AI
    if level == 'basic':
        simplified_basic = []
        for s in sentences:
            simplified = _apply_basic_rules(s)
            simplified = _remove_repetitive_phrases(simplified)
            simplified_basic.append(_normalize_sentence(simplified))
        # Remove duplicates even in basic mode
        simplified_basic = _remove_duplicate_sentences(simplified_basic, similarity_threshold=0.90)
        return '\n'.join(simplified_basic)

    # Intermediate and Advanced: use AI with paragraph grouping
    num_beams = 2 if fast else 4
    batch_size = 8 if fast else 4
    
    # Group sentences into paragraphs (2-3 sentences per paragraph for better context)
    sentences_per_para = 2 if len(sentences) > 4 else 3
    paragraphs = _group_sentences_into_paragraphs(sentences, sentences_per_para)
    
    # Apply basic rules first
    base_paragraphs = [_apply_basic_rules(para) for para in paragraphs]
    
    # First paraphrase pass with "simplify and shorten" prompt
    paraphrased_paragraphs = _paraphrase_batch(
        base_paragraphs,
        num_beams=num_beams,
        batch_size=batch_size,
        prompt_template="simplify and shorten: {text}"
    )
    
    # Advanced level: second paraphrase pass
    if level == 'advanced':
        paraphrased_paragraphs = _paraphrase_batch(
            paraphrased_paragraphs,
            num_beams=num_beams + 1,
            batch_size=max(3, batch_size // 2),
            prompt_template="simplify and shorten further: {text}"
        )
    
    # Split paragraphs back into sentences
    all_simplified_sentences = []
    for para in paraphrased_paragraphs:
        if para.strip():
            para_sentences = _split_paragraph_into_sentences(para)
            all_simplified_sentences.extend(para_sentences)
    
    # If we got fewer sentences than expected, pad with rule-based fallback
    if len(all_simplified_sentences) < len(sentences):
        for idx in range(len(all_simplified_sentences), len(sentences)):
            fallback = _apply_basic_rules(sentences[idx])
            all_simplified_sentences.append(fallback)
    
    # Post-processing: remove repetitive phrases from each sentence
    cleaned_sentences = []
    for sent in all_simplified_sentences:
        cleaned = _remove_repetitive_phrases(sent)
        if cleaned.strip():
            cleaned_sentences.append(_normalize_sentence(cleaned))
    
    # Remove duplicate or near-duplicate sentences (85% similarity threshold)
    unique_sentences = _remove_duplicate_sentences(cleaned_sentences, similarity_threshold=0.85)
    
    # Final normalization
    output = [_normalize_sentence(s) for s in unique_sentences if s.strip()]
    
    return '\n'.join(output) if output else ""


# -----------------------------------------------------
# Example usage
# -----------------------------------------------------
if __name__ == "__main__":
    complex_clause = """
    Notwithstanding anything to the contrary contained herein, the Lessee shall indemnify and hold harmless the Lessor from any liability arising out of the Lessee's use of the premises, including but not limited to, claims of third parties.
    Further, in the event of any dispute, the party seeking relief shall notify the other party and pursue arbitration.
    The parties hereby agree that this agreement shall be governed by the laws of the state.
    The parties hereby acknowledge that any disputes shall be resolved through binding arbitration.
    """

    print("=" * 80)
    print("Testing Improved Simplification System")
    print("=" * 80)
    
    for level in ['basic', 'intermediate', 'advanced']:
        print(f"\n{'='*80}")
        print(f"Level: {level.upper()}")
        print(f"{'='*80}")
        
        start = time.time()
        simplified_clause = simplify_text(complex_clause, level=level, fast=True)
        elapsed = time.time() - start
        
        print(f"\n📝 Original ({len(complex_clause.split())} words):")
        print(complex_clause.strip())
        print(f"\n✨ Simplified ({len(simplified_clause.split())} words):")
        print(simplified_clause)
        print(f"\n⏱  Time: {elapsed:.2f}s | Model: {MODEL_NAME}")
        
        reduction = ((len(complex_clause.split()) - len(simplified_clause.split())) / len(complex_clause.split())) * 100
        print(f"📊 Length reduction: {reduction:.1f}%")