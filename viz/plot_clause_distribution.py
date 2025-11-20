import json
import os
import argparse
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for server environments
import matplotlib.pyplot as plt
from collections import Counter


def load_results(path):
    """Load results JSON in either list or wrapped {results:[...]} format."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and "results" in data:
        return data["results"]
    return data


def calculate_clause_lengths(results):
    """Calculate word count for each clause and convert to buckets."""
    buckets = []

    for clause in results:
        text = clause.get("cleaned") or ""
        word_count = len(text.split())

        # Categorize into buckets
        if word_count <= 20:
            buckets.append("0–20 words")
        elif word_count <= 50:
            buckets.append("21–50 words")
        elif word_count <= 100:
            buckets.append("51–100 words")
        elif word_count <= 150:
            buckets.append("101–150 words")
        else:
            buckets.append("150+ words")

    return Counter(buckets)


def plot_clause_distribution(results, out_path):
    """Generate clause distribution chart from results list."""
    if isinstance(results, list):
        counter = calculate_clause_lengths(results)
    else:
        counter = results
    return plot_clause_length_distribution(counter, out_path)


def plot_clause_length_distribution(counter, out_path):
    labels = list(counter.keys())
    values = list(counter.values())

    colors = plt.get_cmap("Set3").colors[:len(labels)]

    # Donut chart
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        values,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors,
        textprops={'fontsize': 10}
    )

    centre_circle = plt.Circle((0, 0), 0.60, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)

    plt.title("Clause Length Distribution (Word Count)", fontsize=14, fontweight='bold')
    plt.legend(wedges, labels, title="Length Ranges", loc="center left", bbox_to_anchor=(1, 0.6))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved clause length distribution chart to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot clause length distribution pie chart")
    parser.add_argument("results_json", help="Path to results JSON file")
    parser.add_argument("--out-dir", default="uploads", help="Directory to save PNG output")
    args = parser.parse_args()

    results = load_results(args.results_json)
    counter = calculate_clause_lengths(results)

    base = os.path.splitext(os.path.basename(args.results_json))[0]
    out_path = os.path.join(args.out_dir, f"{base}_clause_length_distribution.png")

    os.makedirs(args.out_dir, exist_ok=True)
    plot_clause_length_distribution(counter, out_path)