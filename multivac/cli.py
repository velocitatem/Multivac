"""Command line entry points for Multivac."""

from __future__ import annotations

import argparse
import threading
import json
import sys
from pathlib import Path
from typing import Optional

from .analysis import (
    analyze_codebase,
    classify_stability,
    print_results,
)
from .git_tracker import (
    GitEntropyTracker,
    build_history_summary,
    write_step_summary,
)
from .server import create_app


def run_analyze(args: argparse.Namespace) -> int:
    directory = args.directory.resolve()
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return 1
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        return 1

    results, overall_similarity = analyze_codebase(
        directory,
        max_depth=args.depth,
        verbose=args.verbose,
        min_patterns=args.min_patterns,
    )

    if args.json:
        output = {
            "total_files": len(results),
            "overall_similarity": float(overall_similarity),
            "overall_status": classify_stability(overall_similarity)
            if results
            else "unknown",
            "files": [
                {
                    "path": result.file_path,
                    "avg_similarity_to_others": float(result.avg_similarity_to_others),
                    "pattern_count": result.pattern_count,
                    "unique_patterns": result.unique_patterns,
                    "status": result.status,
                }
                for result in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(
            results,
            overall_similarity,
            args.verbose,
            min_patterns=args.min_patterns,
        )

    return 0


def run_timeline(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    if not repo.exists():
        print(f"Error: Repository path '{repo}' does not exist.")
        return 1

    tracker = GitEntropyTracker(
        repo_path=repo,
        max_depth=args.depth,
        min_patterns=args.min_patterns,
        history_dir=args.history_dir,
    )

    processed = tracker.update_history(recompute=args.recompute)
    if processed:
        print(f"Processed {processed} commit(s).")
    else:
        print("No new commits to process.")

    print(f"History file: {tracker.history_path}")

    if args.summary or args.step_summary:
        history = tracker.load_history()
        summary_text = build_history_summary(
            history,
            fmt=args.summary_format,
            max_rows=max(1, args.summary_rows),
        )
        if args.summary:
            print()
            print(summary_text)
        if args.step_summary:
            if write_step_summary(summary_text):
                print("Appended entropy summary to $GITHUB_STEP_SUMMARY.")
            else:
                print(
                    "Warning: $GITHUB_STEP_SUMMARY is not available; unable to append summary."
                )

    if args.follow:
        try:
            tracker.follow(interval=args.interval)
        except KeyboardInterrupt:
            print("Stopping timeline watcher.")
    return 0


def run_serve(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    if not repo.exists():
        print(f"Error: Repository path '{repo}' does not exist.")
        return 1

    tracker = GitEntropyTracker(
        repo_path=repo,
        max_depth=args.depth,
        min_patterns=args.min_patterns,
        history_dir=args.history_dir,
    )

    initial = tracker.update_history(recompute=False)
    if initial:
        print(f"Processed {initial} commit(s) before starting the server.")
    print(f"History file: {tracker.history_path}")

    stop_event: Optional[threading.Event] = None
    watcher_thread: Optional[threading.Thread] = None

    if args.follow:
        stop_event = threading.Event()
        watcher_thread = threading.Thread(
            target=tracker.follow,
            kwargs={"interval": args.interval, "stop_event": stop_event, "quiet": True},
            daemon=True,
        )
        watcher_thread.start()
        print(
            "Background commit watcher started. New commits will be added automatically."
        )

    app = create_app(tracker)
    try:
        app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    finally:
        if stop_event:
            stop_event.set()
        if watcher_thread:
            watcher_thread.join(timeout=5)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="multivac",
        description="Analyze structural entropy within a git repository over time.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze the current working tree (original behavior).",
    )
    analyze_parser.add_argument(
        "directory",
        type=Path,
        nargs="?",
        default=Path.cwd(),
        help="Directory to analyze (default: current directory)",
    )
    analyze_parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=3,
        help="AST subtree depth (default: 3)",
    )
    analyze_parser.add_argument(
        "-m",
        "--min-patterns",
        type=int,
        default=15,
        help="Minimum number of patterns for a file to be considered substantial.",
    )
    analyze_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show all files, not just chaotic ones",
    )
    analyze_parser.add_argument(
        "-j",
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    analyze_parser.set_defaults(func=run_analyze)

    timeline_parser = subparsers.add_parser(
        "timeline",
        help="Build or update the entropy timeline across git commits.",
    )
    timeline_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the git repository (default: current directory)",
    )
    timeline_parser.add_argument(
        "--history-dir",
        type=Path,
        help="Directory to store entropy artifacts (default: <repo>/.multivac)",
    )
    timeline_parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=3,
        help="AST subtree depth for analysis (default: 3)",
    )
    timeline_parser.add_argument(
        "-m",
        "--min-patterns",
        type=int,
        default=15,
        help="Minimum number of patterns for a file to be considered substantial.",
    )
    timeline_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Keep running and monitor for new commits automatically.",
    )
    timeline_parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=10.0,
        help="Polling interval (seconds) when following for new commits.",
    )
    timeline_parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute the entire timeline from scratch.",
    )
    timeline_parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a short entropy summary after processing.",
    )
    timeline_parser.add_argument(
        "--summary-format",
        choices=["plain", "markdown"],
        default="markdown",
        help="Format for the summary output (default: markdown).",
    )
    timeline_parser.add_argument(
        "--summary-rows",
        type=int,
        default=10,
        help="Number of recent commits to include in the summary (default: 10).",
    )
    timeline_parser.add_argument(
        "--step-summary",
        action="store_true",
        help="Append the summary to $GITHUB_STEP_SUMMARY if set (GitHub Actions).",
    )
    timeline_parser.set_defaults(func=run_timeline)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch a local dashboard to visualize entropy history.",
    )
    serve_parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Path to the git repository (default: current directory)",
    )
    serve_parser.add_argument(
        "--history-dir",
        type=Path,
        help="Directory to store entropy artifacts (default: <repo>/.multivac)",
    )
    serve_parser.add_argument(
        "-d",
        "--depth",
        type=int,
        default=3,
        help="AST subtree depth for analysis (default: 3)",
    )
    serve_parser.add_argument(
        "-m",
        "--min-patterns",
        type=int,
        default=15,
        help="Minimum number of patterns for a file to be considered substantial.",
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface for the dashboard (default: 127.0.0.1)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the dashboard (default: 5000)",
    )
    serve_parser.add_argument(
        "-f",
        "--follow",
        action="store_true",
        help="Watch for new commits while the server is running.",
    )
    serve_parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=10.0,
        help="Polling interval (seconds) when following for new commits.",
    )
    serve_parser.set_defaults(func=run_serve)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    subcommands = {"analyze", "timeline", "serve"}
    if argv and argv[0] not in subcommands and not argv[0].startswith("-"):
        argv = ["analyze", *argv]

    parser = build_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)

    if hasattr(args, "func"):
        return args.func(args)

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
