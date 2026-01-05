#!/usr/bin/env python3
import ast
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import numpy as np
from dataclasses import dataclass
import json


@dataclass
class EntropyResult:
    file_path: str
    avg_similarity_to_others: float
    pattern_count: int
    unique_patterns: int
    status: str  # "stable", "moderate", "chaotic"


def ast_to_structural_hash(node: ast.AST, max_depth: int = 3, current_depth: int = 0) -> List[str]:
    """Extract depth-bounded subtree patterns as canonical hashes."""
    if current_depth >= max_depth:
        return [node.__class__.__name__]

    patterns = [node.__class__.__name__]
    for child in ast.iter_child_nodes(node):
        child_patterns = ast_to_structural_hash(child, max_depth, current_depth + 1)
        patterns.extend([f"{node.__class__.__name__}>{p}" for p in child_patterns])

    return patterns


def parse_file_to_patterns(file_path: Path, max_depth: int = 3) -> List[str]:
    """Parse Python file and extract structural patterns."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        return ast_to_structural_hash(tree, max_depth)
    except (SyntaxError, UnicodeDecodeError):
        return []


def build_distribution(patterns: List[str]) -> Dict[str, float]:
    """Build empirical probability distribution from patterns."""
    if not patterns:
        return {}
    counts = Counter(patterns)
    total = sum(counts.values())
    return {pattern: count / total for pattern, count in counts.items()}


def kl_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Kullback-Leibler divergence D_KL(P || Q)."""
    all_keys = set(p.keys()) | set(q.keys())
    divergence = 0.0
    for key in all_keys:
        p_val = p.get(key, 1e-10)
        q_val = q.get(key, 1e-10)
        if p_val > 0:
            divergence += p_val * np.log2(p_val / q_val)
    return divergence


def jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    m = {key: 0.5 * (p.get(key, 0) + q.get(key, 0)) for key in all_keys}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jsd_to_similarity(jsd: float) -> float:
    """Convert JSD to similarity score [0,1]."""
    return 1.0 - jsd


def classify_stability(similarity: float) -> str:
    """Classify code stability based on pairwise JSD similarity."""
    if similarity >= 0.65:
        return "stable"
    elif similarity >= 0.45:
        return "moderate"
    else:
        return "chaotic"


def collect_python_files(directory: Path) -> List[Path]:
    """Recursively collect all .py files in directory, excluding venv directories."""
    exclude_dirs = {'venv', '.venv', '__pycache__', '.git', 'node_modules'}
    return [
        p for p in directory.rglob("*.py")
        if p.is_file() and not p.name.startswith('.') and not any(part in exclude_dirs for part in p.parts)
    ]


def analyze_codebase(directory: Path, max_depth: int = 3, verbose: bool = False, min_patterns: int = 15) -> Tuple[List[EntropyResult], float]:
    """Analyze codebase structural entropy using pairwise file comparisons."""
    py_files = collect_python_files(directory)

    if not py_files:
        return [], 0.0

    # Extract patterns from all files
    file_patterns = {f: parse_file_to_patterns(f, max_depth) for f in py_files}
    file_distributions = {f: build_distribution(patterns) for f, patterns in file_patterns.items() if patterns}

    if not file_distributions:
        return [], 0.0

    if len(file_distributions) == 1:
        # Single file - no comparison possible
        single_file = list(file_distributions.keys())[0]
        return [EntropyResult(
            file_path=str(single_file.relative_to(directory)),
            avg_similarity_to_others=1.0,
            pattern_count=len(file_patterns[single_file]),
            unique_patterns=len(file_distributions[single_file]),
            status="stable"
        )], 1.0

    # Separate substantial files from tiny ones
    substantial_files = {f: d for f, d in file_distributions.items() if len(file_patterns[f]) >= min_patterns}

    if len(substantial_files) < 2:
        # Not enough substantial files to compare
        files_to_use = file_distributions
    else:
        files_to_use = substantial_files

    # Compute pairwise similarities between files
    files = list(files_to_use.keys())
    file_similarities = {}

    for i, file_a in enumerate(files):
        similarities = []
        for j, file_b in enumerate(files):
            if i != j:
                jsd = jensen_shannon_divergence(files_to_use[file_a], files_to_use[file_b])
                similarities.append(jsd_to_similarity(jsd))
        file_similarities[file_a] = np.mean(similarities) if similarities else 1.0

    # For small files, compare to substantial files only
    for file_path in file_distributions:
        if file_path not in file_similarities:
            similarities = []
            for substantial_file in files:
                if file_path != substantial_file:
                    jsd = jensen_shannon_divergence(file_distributions[file_path], files_to_use[substantial_file])
                    similarities.append(jsd_to_similarity(jsd))
            file_similarities[file_path] = np.mean(similarities) if similarities else 1.0

    # Build results
    results = []
    for file_path, avg_sim in file_similarities.items():
        results.append(EntropyResult(
            file_path=str(file_path.relative_to(directory)),
            avg_similarity_to_others=avg_sim,
            pattern_count=len(file_patterns[file_path]),
            unique_patterns=len(file_distributions[file_path]),
            status=classify_stability(avg_sim)
        ))

    # Overall codebase similarity using median of substantial files to reduce outlier impact
    substantial_sims = [sim for f, sim in file_similarities.items() if f in substantial_files]
    overall_similarity = np.median(substantial_sims) if substantial_sims else np.median(list(file_similarities.values()))

    return sorted(results, key=lambda x: x.avg_similarity_to_others), overall_similarity


def print_results(results: List[EntropyResult], overall_similarity: float, verbose: bool, min_patterns: int = 15):
    """Print analysis results."""
    if not results:
        print("No Python files found or all files failed to parse.")
        return

    overall_status = classify_stability(overall_similarity)
    status_counts = Counter(r.status for r in results)
    chaotic_files = [r for r in results if r.status == "chaotic"]
    substantial_files = [r for r in results if r.pattern_count >= min_patterns]

    print(f"\n{'='*60}")
    print(f"CODEBASE ENTROPY ANALYSIS")
    print(f"{'='*60}\n")
    print(f"Total files analyzed: {len(results)}")
    print(f"  Substantial files (≥{min_patterns} patterns): {len(substantial_files)}")
    print(f"  Small files (<{min_patterns} patterns): {len(results) - len(substantial_files)}")
    print(f"Median pairwise similarity: {overall_similarity:.4f}")
    print(f"Overall status: {overall_status.upper()}\n")

    print(f"Status distribution:")
    print(f"  Stable (≥0.65):   {status_counts['stable']:3d} files")
    print(f"  Moderate (≥0.45): {status_counts['moderate']:3d} files")
    print(f"  Chaotic (<0.45):  {status_counts['chaotic']:3d} files\n")

    if chaotic_files:
        print(f"{'='*60}")
        print(f"OUTLIER FILES (avg similarity to others < 0.45):")
        print(f"{'='*60}\n")
        for r in chaotic_files[:10]:  # Show top 10
            print(f"{r.file_path}")
            print(f"  Avg similarity: {r.avg_similarity_to_others:.4f} | Patterns: {r.pattern_count} | Unique: {r.unique_patterns}")
            print()

    if verbose:
        print(f"\n{'='*60}")
        print(f"ALL FILES (sorted by similarity, ascending):")
        print(f"{'='*60}\n")
        for r in results:
            status_symbol = {"stable": "✓", "moderate": "~", "chaotic": "✗"}[r.status]
            print(f"[{status_symbol}] {r.file_path}")
            print(f"    Avg similarity: {r.avg_similarity_to_others:.4f} | Patterns: {r.pattern_count}")

    print(f"\n{'='*60}")
    print(f"INTERPRETATION:")
    print(f"{'='*60}\n")

    if overall_status == "stable":
        print("Your codebase shows high structural consistency. Files share similar")
        print("AST patterns, indicating disciplined, uniform development practices.")
    elif overall_status == "moderate":
        print("Your codebase has moderate structural variance. Some outlier files")
        print("diverge from the structural norm. Review and refactor for consistency.")
    else:
        print("Your codebase exhibits high structural entropy - classic 'vibe-coding'")
        print("chaos. Files show significant structural divergence, suggesting each")
        print("was generated with different patterns. Refactor for consistency.")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze codebase structural entropy to detect vibe-coding chaos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s .                    # Analyze current directory
  %(prog)s /path/to/project     # Analyze specific project
  %(prog)s . -v                 # Verbose output with all files
  %(prog)s . --depth 4          # Deeper AST analysis
        """
    )
    parser.add_argument("directory", type=Path, help="Directory to analyze")
    parser.add_argument("-d", "--depth", type=int, default=3, help="AST subtree depth (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show all files, not just chaotic ones")
    parser.add_argument("-j", "--json", action="store_true", help="Output results as JSON")

    args = parser.parse_args()

    if not args.directory.exists():
        print(f"Error: Directory '{args.directory}' does not exist.")
        return 1

    if not args.directory.is_dir():
        print(f"Error: '{args.directory}' is not a directory.")
        return 1

    results, overall_similarity = analyze_codebase(args.directory, args.depth, args.verbose)

    if args.json:
        output = {
            "total_files": len(results),
            "overall_similarity": float(overall_similarity),
            "overall_status": classify_stability(overall_similarity) if results else "unknown",
            "files": [
                {
                    "path": r.file_path,
                    "avg_similarity_to_others": r.avg_similarity_to_others,
                    "pattern_count": r.pattern_count,
                    "unique_patterns": r.unique_patterns,
                    "status": r.status
                }
                for r in results
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(results, overall_similarity, args.verbose)

    return 0


if __name__ == "__main__":
    exit(main())
