"""Flask dashboard for the Multivac entropy timeline."""

from __future__ import annotations

from typing import Any

from flask import Flask, jsonify, render_template_string

from .git_tracker import GitEntropyTracker

INDEX_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Multivac Entropy Timeline</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background: #f8f9fa; color: #212529; }
    h1 { margin-bottom: 0.5rem; }
    #summary { margin-bottom: 1rem; font-size: 0.95rem; }
    canvas { width: 100%; max-width: 960px; height: 360px; background: #fff; border: 1px solid #dee2e6; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    table { border-collapse: collapse; width: 100%; max-width: 960px; margin-top: 1.25rem; background: #fff; border: 1px solid #dee2e6; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid #eee; text-align: left; font-size: 0.9rem; }
    th { background: #f1f3f5; font-weight: 600; }
    tr:last-child td { border-bottom: none; }
    .status-stable { color: #2b8a3e; font-weight: 600; }
    .status-moderate { color: #e67700; font-weight: 600; }
    .status-chaotic { color: #d6336c; font-weight: 600; }
  </style>
</head>
<body>
  <h1>Multivac Entropy Timeline</h1>
  <p id=\"summary\">Loading entropy history…</p>
  <canvas id=\"entropy-canvas\" width=\"960\" height=\"360\"></canvas>
  <table>
    <thead>
      <tr>
        <th>Commit</th>
        <th>Date</th>
        <th>Similarity</th>
        <th>Status</th>
        <th>Files</th>
        <th>Chaotic Hotspots</th>
      </tr>
    </thead>
    <tbody id=\"history-table-body\"></tbody>
  </table>
  {% raw %}
  <script>
    async function refreshHistory() {
      try {
        const response = await fetch('/api/history');
        if (!response.ok) {
          console.warn('Failed to fetch entropy history:', response.statusText);
          return;
        }
        const payload = await response.json();
        renderHistory(payload.commits || []);
      } catch (err) {
        console.warn('Unable to refresh history', err);
      }
    }

    function renderHistory(commits) {
      const summaryEl = document.getElementById('summary');
      const canvas = document.getElementById('entropy-canvas');
      const tbody = document.getElementById('history-table-body');
      const ctx = canvas.getContext('2d');
      const width = canvas.width;
      const height = canvas.height;
      ctx.clearRect(0, 0, width, height);
      tbody.innerHTML = '';

      if (!commits.length) {
        summaryEl.textContent = 'No commits analyzed yet.';
        return;
      }

      const latest = commits[commits.length - 1];
      summaryEl.textContent = `Total commits analyzed: ${commits.length}. Latest status: ${(latest.overall_status || 'unknown').toUpperCase()} at similarity ${(latest.overall_similarity || 0).toFixed(3)}.`;

      const padding = 48;
      const plotWidth = width - padding * 2;
      const plotHeight = height - padding * 2;

      ctx.strokeStyle = '#adb5bd';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(padding, padding);
      ctx.lineTo(padding, height - padding);
      ctx.lineTo(width - padding, height - padding);
      ctx.stroke();

      ctx.fillStyle = '#495057';
      ctx.font = '12px sans-serif';
      ctx.fillText('Similarity', padding - 8, padding - 16);
      ctx.fillText('Commits', width - padding - 40, height - padding + 28);
      ctx.fillText('1.0', padding - 36, padding + 4);
      ctx.fillText('0.0', padding - 36, height - padding + 4);

      ctx.beginPath();
      ctx.strokeStyle = '#2b8a3e';
      ctx.lineWidth = 2;

      commits.forEach((entry, index) => {
        const normalized = Math.max(0, Math.min(1, entry.overall_similarity || 0));
        const x = padding + (commits.length === 1 ? 0 : (index / (commits.length - 1)) * plotWidth);
        const y = padding + (1 - normalized) * plotHeight;
        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });
      ctx.stroke();

      commits.forEach((entry, index) => {
        const normalized = Math.max(0, Math.min(1, entry.overall_similarity || 0));
        const x = padding + (commits.length === 1 ? 0 : (index / (commits.length - 1)) * plotWidth);
        const y = padding + (1 - normalized) * plotHeight;
        ctx.fillStyle = '#2b8a3e';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
      });

      commits
        .slice()
        .reverse()
        .forEach((entry) => {
          const row = document.createElement('tr');
          const chaoticPaths = (entry.worst_files || [])
            .map((f) => `${f.path} (${Number(f.avg_similarity || 0).toFixed(2)})`)
            .join(', ');
          const statusClass = `status-${(entry.overall_status || 'unknown').toLowerCase()}`;
          row.innerHTML = `
            <td><code>${entry.commit.slice(0, 7)}</code></td>
            <td>${new Date((entry.timestamp || 0) * 1000).toLocaleString()}</td>
            <td>${Number(entry.overall_similarity || 0).toFixed(3)}</td>
            <td class="${statusClass}">${(entry.overall_status || 'unknown').toUpperCase()}</td>
            <td>${entry.total_files}</td>
            <td>${chaoticPaths}</td>
          `;
          tbody.appendChild(row);
        });
    }

    refreshHistory();
    setInterval(refreshHistory, 10000);
  </script>
  {% endraw %}
</body>
</html>
"""


def create_app(tracker: GitEntropyTracker) -> Flask:
    app = Flask(__name__)

    @app.route("/api/history")
    def history_endpoint() -> Any:
        history = tracker.load_history()
        history.setdefault("commits", [])
        return jsonify(history)

    @app.route("/")
    def index() -> Any:
        return render_template_string(INDEX_TEMPLATE)

    return app


__all__ = ["create_app", "INDEX_TEMPLATE"]
