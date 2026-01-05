"""Git-driven entropy timeline tracking."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from .analysis import EXCLUDE_DIRS, analyze_file_sources, classify_stability

HISTORY_VERSION = 2
DEFAULT_HISTORY_SUBDIR = ".multivac"


class GitEntropyTracker:
    """Compute and persist entropy measurements across git commits."""

    def __init__(
        self,
        repo_path: Path,
        max_depth: int = 3,
        min_patterns: int = 15,
        history_dir: Optional[Path] = None,
    ) -> None:
        self.repo_path = repo_path
        self.max_depth = max_depth
        self.min_patterns = min_patterns
        if history_dir is None:
            history_dir = repo_path / DEFAULT_HISTORY_SUBDIR
        elif not history_dir.is_absolute():
            history_dir = repo_path / history_dir
        self.history_dir = history_dir
        self.history_path = self.history_dir / "entropy_history.json"
        self.lock = threading.RLock()

    def _git(self, args: List[str]) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_path,
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Git command failed: git {' '.join(args)}\n{result.stderr.strip()}"
            )
        return result.stdout

    def list_commits(self) -> List[str]:
        output = self._git(["log", "--pretty=%H", "--reverse", "HEAD"])
        return [line.strip() for line in output.splitlines() if line.strip()]

    def load_history(self) -> Dict[str, Any]:
        with self.lock:
            return self._load_history_unlocked()

    def _load_history_unlocked(self) -> Dict[str, Any]:
        if not self.history_path.exists():
            return {"version": HISTORY_VERSION, "commits": []}
        try:
            with self.history_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {"version": HISTORY_VERSION, "commits": []}
        data.setdefault("version", HISTORY_VERSION)
        data.setdefault("commits", [])
        return data

    def save_history(self, history: Dict[str, Any]) -> None:
        with self.lock:
            history = dict(history)
            history["version"] = HISTORY_VERSION
            self.history_dir.mkdir(parents=True, exist_ok=True)
            temp_path = self.history_path.with_suffix(".tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            temp_path.replace(self.history_path)

    def get_python_file_sources(self, commit: str) -> Dict[str, str]:
        output = self._git(["ls-tree", "-r", "--full-tree", commit, "--name-only"])
        file_sources: Dict[str, str] = {}
        for line in output.splitlines():
            path = line.strip()
            if not path or not path.endswith(".py"):
                continue
            path_obj = Path(path)
            if path_obj.name.startswith('.') or any(
                part in EXCLUDE_DIRS for part in path_obj.parts
            ):
                continue
            try:
                source = self._git(["show", f"{commit}:{path}"])
            except RuntimeError as exc:
                print(
                    f"Warning: unable to read {path} in {commit[:10]}: {exc}",
                    file=sys.stderr,
                )
                continue
            file_sources[path] = source
        return file_sources

    def compute_commit_snapshot(self, commit: str) -> Dict[str, Any]:
        meta_output = self._git([
            "show",
            "-s",
            "--format=%H%x1f%ct%x1f%an%x1f%s",
            commit,
        ]).strip()
        parts = meta_output.split("\x1f")
        if len(parts) != 4:
            raise RuntimeError(f"Unexpected git metadata format for commit {commit}")
        commit_hash, timestamp_str, author, summary = parts
        try:
            timestamp = int(timestamp_str)
        except ValueError as exc:
            raise RuntimeError(f"Invalid commit timestamp for {commit}") from exc

        file_sources = self.get_python_file_sources(commit)
        results, overall_similarity = analyze_file_sources(
            file_sources,
            max_depth=self.max_depth,
            min_patterns=self.min_patterns,
        )

        status_counts = Counter(result.status for result in results)
        chaotic_files = [result for result in results if result.status == "chaotic"]
        chaotic_files.sort(key=lambda result: result.avg_similarity_to_others)
        worst_files = [
            {
                "path": result.file_path,
                "avg_similarity": float(result.avg_similarity_to_others),
                "pattern_count": result.pattern_count,
                "unique_patterns": result.unique_patterns,
            }
            for result in chaotic_files[:5]
        ]

        if results:
            overall_status = classify_stability(overall_similarity)
            similarity_value = float(overall_similarity)
        else:
            overall_status = "unknown"
            similarity_value = 0.0

        return {
            "commit": commit_hash,
            "timestamp": timestamp,
            "author": author,
            "summary": summary,
            "overall_similarity": similarity_value,
            "overall_status": overall_status,
            "total_files": len(results),
            "substantial_files": sum(
                1 for result in results if result.pattern_count >= self.min_patterns
            ),
            "status_counts": {
                "stable": status_counts.get("stable", 0),
                "moderate": status_counts.get("moderate", 0),
                "chaotic": status_counts.get("chaotic", 0),
            },
            "max_depth": self.max_depth,
            "min_patterns": self.min_patterns,
            "worst_files": worst_files,
        }

    def update_history(self, recompute: bool = False) -> int:
        commits = self.list_commits()
        if not commits:
            print("No commits found in repository.")
            return 0

        if recompute:
            history = {"version": HISTORY_VERSION, "commits": []}
            existing_map: Dict[str, Dict[str, Any]] = {}
            target_commits = commits
        else:
            history = self.load_history()
            existing_map = {entry["commit"]: entry for entry in history["commits"]}
            target_commits = [commit for commit in commits if commit not in existing_map]

        new_records: List[Dict[str, Any]] = []
        for commit in target_commits:
            try:
                record = self.compute_commit_snapshot(commit)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"Failed to compute entropy for commit {commit[:10]}: {exc}",
                    file=sys.stderr,
                )
                continue
            new_records.append(record)

        if not new_records and not recompute:
            return 0

        if recompute:
            existing_map = {}
        else:
            existing_map = {entry["commit"]: entry for entry in history["commits"]}

        for record in new_records:
            existing_map[record["commit"]] = record

        ordered_commits = [commit for commit in commits if commit in existing_map]
        history["commits"] = [existing_map[commit] for commit in ordered_commits]
        history["version"] = HISTORY_VERSION
        self.save_history(history)
        return len(new_records)

    def follow(
        self,
        interval: float = 10.0,
        stop_event: Optional[threading.Event] = None,
        quiet: bool = False,
    ) -> None:
        if not quiet:
            print(
                f"Watching {self.repo_path} for new commits (interval={interval:.1f}s)."
            )
        try:
            while True:
                processed = self.update_history()
                if processed and not quiet:
                    print(f"Recorded {processed} new commit(s).")
                if stop_event:
                    if stop_event.wait(interval):
                        break
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            if not quiet:
                print("Stopping commit watcher.")


def build_history_summary(
    history: Dict[str, Any], fmt: str = "markdown", max_rows: int = 10
) -> str:
    commits = history.get("commits") or []
    if not commits:
        return "No commits analyzed yet. Run `multivac timeline` to populate the history."

    total = len(commits)
    latest = commits[-1]
    statuses = Counter(entry.get("overall_status", "unknown") for entry in commits)
    stable = statuses.get("stable", 0)
    moderate = statuses.get("moderate", 0)
    chaotic = statuses.get("chaotic", 0)

    recent = commits[-max_rows:]

    def describe_hotspots(entry: Dict[str, Any]) -> str:
        hotspots = entry.get("worst_files") or []
        if not hotspots:
            return "—"
        trimmed = hotspots[:3]
        return ", ".join(
            f"{item['path']} ({item.get('avg_similarity', 0.0):.2f})" for item in trimmed
        )

    if fmt == "plain":
        lines = [
            "Multivac Entropy Timeline",
            f"Total commits analyzed: {total}",
            f"Latest commit {latest['commit'][:7]} ({latest.get('overall_status', 'unknown').upper()})",
            f"Similarity: {latest.get('overall_similarity', 0.0):.3f} | Files analyzed: {latest.get('total_files', 0)}",
            f"Stable: {stable}  Moderate: {moderate}  Chaotic: {chaotic}",
            "",
            "Recent commits:",
        ]
        for entry in reversed(recent):
            lines.append(
                f"- {entry['commit'][:7]} | {entry.get('overall_status', 'unknown').upper()} | "
                f"sim={entry.get('overall_similarity', 0.0):.3f} | {describe_hotspots(entry)}"
            )
        return "\n".join(lines)

    lines = [
        "## Multivac Entropy Timeline",
        "",
        f"- Total commits analyzed: **{total}**",
        f"- Latest: `{latest['commit'][:7]}` ({latest.get('overall_status', 'unknown').upper()}) at {latest.get('overall_similarity', 0.0):.3f}",
        f"- Files analyzed: {latest.get('total_files', 0)} | Stable: {stable} · Moderate: {moderate} · Chaotic: {chaotic}",
        "",
        "| Commit | Similarity | Status | Files | Chaotic hotspots |",
        "| --- | --- | --- | --- | --- |",
    ]
    for entry in reversed(recent):
        lines.append(
            "| `{commit}` | {similarity:.3f} | {status} | {files} | {hotspots} |".format(
                commit=entry["commit"][:7],
                similarity=entry.get("overall_similarity", 0.0),
                status=entry.get("overall_status", "unknown").upper(),
                files=entry.get("total_files", 0),
                hotspots=describe_hotspots(entry),
            )
        )
    return "\n".join(lines)


def write_step_summary(summary: str) -> bool:
    path_str = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path_str:
        return False
    target = Path(path_str)
    try:
        with target.open("a", encoding="utf-8") as handle:
            handle.write(summary)
            if not summary.endswith("\n"):
                handle.write("\n")
    except OSError:
        return False
    return True


__all__ = [
    "DEFAULT_HISTORY_SUBDIR",
    "GitEntropyTracker",
    "HISTORY_VERSION",
    "build_history_summary",
    "write_step_summary",
]
