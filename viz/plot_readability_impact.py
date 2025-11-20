import json
import os
import argparse
import re
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------
# Gunning Fog Index Functions
# ---------------------------

def count_complex_words(text):
    """
    Complex word = 3 or more syllables.
    This is a simplified heuristic.
    """
    words = re.findall(r"\w+", text.lower())
    count = 0
    for w in words:
        syllables = len(re.findall(r"[aeiouy]+", w))
        if syllables >= 3:
            count += 1
    return count


def gunning_fog(text):
    """Compute Fog Index = 0.4 * ( (words/sentences) + %complex_words*100 )."""
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"\w+", text)

    if len(words) == 0 or len(sentences) == 0:
        return 0

    wc = len(words)
    sc = len(sentences)
    complex_words = count_complex_words(text)

    percent_complex = complex_words / wc if wc else 0

    score = 0.4 * ((wc / sc) + (percent_complex * 100))
    return round(score, 2)

# ---------------------------
# Load results
# ---------------------------

def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["results"] if isinstance(data, dict) and "results" in data else data


# ---------------------------
# Plotting Function
# ---------------------------

def plot_complexity(results, out_path):
    clause_labels = []
    before_scores = []
    after_scores = []

    for i, r in enumerate(results[:12]):  # Limit to first 12 for clean chart
        cleaned = r.get("cleaned", "")
        simple = r.get("simple", "")

        before = gunning_fog(cleaned)
        after = gunning_fog(simple)

        clause_labels.append(f"Clause {i+1}")
        before_scores.append(before)
        after_scores.append(after)

    x = np.arange(len(clause_labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, before_scores, width, label="Before", color="#ff6384")
    plt.bar(x + width/2, after_scores, width, label="After", color="#36a2eb")

    plt.xticks(x, clause_labels, rotation=45)
    plt.ylabel("Gunning Fog Index (lower = easier)")
    plt.title("Sentence Complexity Score: Before vs After Simplification")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved Sentence Complexity Chart to: {out_path}")


def plot_readability(results, out_path):
    """Compatibility wrapper: application expects `plot_readability(results, out_path)`.
    Call the sentence complexity plot (Gunning Fog) as a readability proxy.
    """
    return plot_complexity(results, out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot sentence complexity using Gunning Fog Index")
    parser.add_argument("results_json", help="Path to results JSON")
    parser.add_argument("--out-dir", default="uploads", help="Where to save output")
    args = parser.parse_args()

    results = load_results(args.results_json)
    base = os.path.splitext(os.path.basename(args.results_json))[0]

    out_path = os.path.join(args.out_dir, f"{base}_sentence_complexity.png")
    os.makedirs(args.out_dir, exist_ok=True)

    plot_complexity(results, out_path)
