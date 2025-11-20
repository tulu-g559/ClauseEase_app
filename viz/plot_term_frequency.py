"""
plot_term_frequency.py

Creates a horizontal bar chart showing top N legal terms by frequency.

Usage:
    python viz/plot_term_frequency.py path/to/file.results.json --top 10 --out-dir uploads

Saves: <basename>_top_terms.png in out-dir.
"""
import json
import os
import argparse
from collections import Counter

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception as e:
    raise RuntimeError("matplotlib and seaborn are required. Install with: pip install matplotlib seaborn")

sns.set(style="whitegrid")


def load_results(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and 'results' in data:
        return data['results']
    return data


def extract_terms(results):
    counts = Counter()
    for r in results:
        terms = r.get('terms') or {}
        if isinstance(terms, dict):
            for t in terms.keys():
                counts[t] += 1
        elif isinstance(terms, (list, tuple)):
            for t in terms:
                counts[t] += 1
        # otherwise ignore
    return counts


def plot_top_terms(results, top_n, out_path):
    counts = extract_terms(results)
    if not counts:
        print('No terms found in results; nothing to plot.')
        return

    top = counts.most_common(top_n)
    labels = [t for t,_ in reversed(top)]
    values = [v for _,v in reversed(top)]

    plt.figure(figsize=(8, max(4, len(labels)*0.5)))
    sns.barplot(x=values, y=labels, palette='Blues_d')
    plt.title(f'Top {top_n} Legal Terms')
    plt.xlabel('Frequency')
    plt.ylabel('Term')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved term frequency chart to: {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot top legal terms')
    parser.add_argument('results_json', help='Path to results json file')
    parser.add_argument('--top', type=int, default=10, help='Number of top terms to display')
    parser.add_argument('--out-dir', default='uploads', help='Directory to save PNG')
    args = parser.parse_args()

    results = load_results(args.results_json)
    base = os.path.splitext(os.path.basename(args.results_json))[0]
    out_path = os.path.join(args.out_dir, f"{base}_top_terms.png")
    os.makedirs(args.out_dir, exist_ok=True)
    plot_top_terms(results, args.top, out_path)
