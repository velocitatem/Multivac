"""Core AST-based structural entropy analysis utilities."""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

EXCLUDE_DIRS = {"venv", ".venv", "__pycache__", ".git", "node_modules"}


@dataclass
class EntropyResult:
    """Per-file entropy metrics."""

    file_path: str
    avg_similarity_to_others: float
    pattern_count: int
    unique_patterns: int
    status: str  # "stable", "moderate", "chaotic"


def ast_to_structural_hash(
    node: ast.AST,
    max_depth: int = 3,
    current_depth: int = 0,
) -> List[str]:
    """Extract depth-bounded subtree patterns as canonical hashes."""
    if current_depth >= max_depth:
        return [node.__class__.__name__]

    patterns = [node.__class__.__name__]
    for child in ast.iter_child_nodes(node):
        child_patterns = ast_to_structural_hash(child, max_depth, current_depth + 1)
        patterns.extend([f"{node.__class__.__name__}>{p}" for p in child_patterns])

    return patterns


def parse_source_to_patterns(source: str, label: str, max_depth: int = 3) -> List[str]:
    """Parse Python source into structural patterns."""
    try:
        tree = ast.parse(source, filename=label)
    except (SyntaxError, UnicodeDecodeError):
        return []
    return ast_to_structural_hash(tree, max_depth)


def parse_file_to_patterns(file_path: Path, max_depth: int = 3) -> List[str]:
    """Parse Python file and extract structural patterns."""
    try:
        source = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    return parse_source_to_patterns(source, str(file_path), max_depth)


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
    return float(divergence)


def jensen_shannon_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    all_keys = set(p.keys()) | set(q.keys())
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in all_keys}
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jsd_to_similarity(jsd: float) -> float:
    """Convert JSD to similarity score [0,1]."""
    similarity = 1.0 - jsd
    if similarity < 0.0:
        return 0.0
    if similarity > 1.0:
        return 1.0
    return similarity


def classify_stability(similarity: float) -> str:
    """Classify code stability based on pairwise similarity."""
    if similarity >= 0.65:
        return "stable"
    if similarity >= 0.45:
        return "moderate"
    return "chaotic"


def collect_python_files(directory: Path) -> List[Path]:
    """Recursively collect python files, skipping common virtualenv folders."""
    return [
        path
        for path in directory.rglob("*.py")
        if path.is_file()
        and not path.name.startswith('.')
        and not any(part in EXCLUDE_DIRS for part in path.parts)
    ]


def analyze_file_sources(
    file_sources: Dict[str, str],
    max_depth: int = 3,
    min_patterns: int = 15,
) -> Tuple[List[EntropyResult], float]:
    """Analyze structural entropy for provided python sources."""
    if not file_sources:
        return [], 0.0

    file_patterns: Dict[str, List[str]] = {}
    for path, source in file_sources.items():
        patterns = parse_source_to_patterns(source, path, max_depth)
        if patterns:
            file_patterns[path] = patterns

    if not file_patterns:
        return [], 0.0

    file_distributions: Dict[str, Dict[str, float]] = {
        path: build_distribution(patterns)
        for path, patterns in file_patterns.items()
        if patterns
    }

    if not file_distributions:
        return [], 0.0

    if len(file_distributions) == 1:
        single_path = next(iter(file_distributions))
        patterns = file_patterns[single_path]
        distribution = file_distributions[single_path]
        result = EntropyResult(
            file_path=single_path,
            avg_similarity_to_others=1.0,
            pattern_count=len(patterns),
            unique_patterns=len(distribution),
            status="stable",
        )
        return [result], 1.0

    substantial_files = {
        path: dist
        for path, dist in file_distributions.items()
        if len(file_patterns[path]) >= min_patterns
    }

    files_to_use = file_distributions if len(substantial_files) < 2 else substantial_files

    files = list(files_to_use.keys())
    file_similarities: Dict[str, float] = {}

    for i, file_a in enumerate(files):
        similarities: List[float] = []
        for j, file_b in enumerate(files):
            if i == j:
                continue
            jsd = jensen_shannon_divergence(files_to_use[file_a], files_to_use[file_b])
            similarities.append(jsd_to_similarity(jsd))
        file_similarities[file_a] = float(np.mean(similarities)) if similarities else 1.0

    for file_path in file_distributions:
        if file_path in file_similarities:
            continue
        similarities = []
        for substantial_file in files:
            if file_path == substantial_file:
                continue
            jsd = jensen_shannon_divergence(
                file_distributions[file_path],
                files_to_use[substantial_file],
            )
            similarities.append(jsd_to_similarity(jsd))
        file_similarities[file_path] = float(np.mean(similarities)) if similarities else 1.0

    results: List[EntropyResult] = []
    for file_path, avg_sim in file_similarities.items():
        distribution = file_distributions[file_path]
        patterns = file_patterns[file_path]
        results.append(
            EntropyResult(
                file_path=file_path,
                avg_similarity_to_others=float(avg_sim),
                pattern_count=len(patterns),
                unique_patterns=len(distribution),
                status=classify_stability(float(avg_sim)),
            )
        )

    substantial_sims = [
        sim for path, sim in file_similarities.items() if path in substantial_files
    ]
    if substantial_sims:
        overall_similarity = float(np.median(substantial_sims))
    else:
        overall_similarity = float(np.median(list(file_similarities.values())))

    return sorted(results, key=lambda x: x.avg_similarity_to_others), overall_similarity


def analyze_codebase(
    directory: Path,
    max_depth: int = 3,
    verbose: bool = False,
    min_patterns: int = 15,
) -> Tuple[List[EntropyResult], float]:
    """Analyze codebase structural entropy using pairwise comparisons."""
    python_files = collect_python_files(directory)

    if not python_files:
        return [], 0.0

    file_sources: Dict[str, str] = {}
    for path in python_files:
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        relative_path = str(path.relative_to(directory))
        file_sources[relative_path] = source

    return analyze_file_sources(
        file_sources,
        max_depth=max_depth,
        min_patterns=min_patterns,
    )


def print_results(
    results: List[EntropyResult],
    overall_similarity: float,
    verbose: bool,
    min_patterns: int = 15,
) -> None:
    """Print analysis results in a human-friendly format."""
    if not results:
        print("No Python files found or all files failed to parse.")
        return

    overall_status = classify_stability(overall_similarity)
    status_counts = Counter(r.status for r in results)
    chaotic_files = [r for r in results if r.status == "chaotic"]
    substantial_files = [r for r in results if r.pattern_count >= min_patterns]

    print(f"\n{'=' * 60}")
    print("CODEBASE ENTROPY ANALYSIS")
    print(f"{'=' * 60}\n")
    print(f"Total files analyzed: {len(results)}")
    print(f"  Substantial files (≥{min_patterns} patterns): {len(substantial_files)}")
    print(f"  Small files (<{min_patterns} patterns): {len(results) - len(substantial_files)}")
    print(f"Median pairwise similarity: {overall_similarity:.4f}")
    print(f"Overall status: {overall_status.upper()}\n")

    print("Status distribution:")
    print(f"  Stable (≥0.65):   {status_counts['stable']:3d} files")
    print(f"  Moderate (≥0.45): {status_counts['moderate']:3d} files")
    print(f"  Chaotic (<0.45):  {status_counts['chaotic']:3d} files\n")

    if chaotic_files:
        print(f"{'=' * 60}")
        print("OUTLIER FILES (avg similarity to others < 0.45):")
        print(f"{'=' * 60}\n")
        for result in chaotic_files[:10]:
            print(result.file_path)
            print(
                f"  Avg similarity: {result.avg_similarity_to_others:.4f} | "
                f"Patterns: {result.pattern_count} | Unique: {result.unique_patterns}"
            )
            print()

    if verbose:
        print(f"\n{'=' * 60}")
        print("ALL FILES (sorted by similarity, ascending):")
        print(f"{'=' * 60}\n")
        for result in results:
            status_symbol = {"stable": "✓", "moderate": "~", "chaotic": "✗"}[result.status]
            print(f"[{status_symbol}] {result.file_path}")
            print(
                f"    Avg similarity: {result.avg_similarity_to_others:.4f} | "
                f"Patterns: {result.pattern_count}"
            )

    print(f"\n{'=' * 60}")
    print("INTERPRETATION:")
    print(f"{'=' * 60}\n")

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


__all__ = [
    "EntropyResult",
    "EXCLUDE_DIRS",
    "analyze_codebase",
    "analyze_file_sources",
    "build_distribution",
    "classify_stability",
    "collect_python_files",
    "print_results",
]
