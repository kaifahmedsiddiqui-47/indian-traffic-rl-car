"""Shared helpers for turning episode results into a test report.

An episode result is a dict with the keys: scenario, reward, steps, outcome, speed (average),
auto_brakes, violations. `outcome` is one of: destination, collision, double_line, stuck, timeout.
"""

import csv
import json
import os

import numpy as np

CSV_FIELDS = [
    "scenario", "episodes", "success_pct", "collision_pct", "double_line_pct", "timeout_pct",
    "avg_reward", "avg_steps", "avg_speed", "avg_auto_brakes",
]


def summarize(scenario, results):
    """Aggregate a list of episode results of one scenario."""
    count = len(results)
    if count == 0:
        return {
            "scenario": scenario, "episodes": 0, "success_pct": 0.0, "collision_pct": 0.0,
            "double_line_pct": 0.0, "timeout_pct": 0.0, "avg_reward": 0.0, "avg_steps": 0.0,
            "avg_speed": 0.0, "avg_auto_brakes": 0.0,
        }

    def share(outcomes):
        return 100.0 * sum(1 for r in results if r["outcome"] in outcomes) / count

    return {
        "scenario": scenario,
        "episodes": count,
        "success_pct": share(("destination",)),
        "collision_pct": share(("collision",)),
        "double_line_pct": share(("double_line",)),
        "timeout_pct": share(("stuck", "timeout")),
        "avg_reward": float(np.mean([r["reward"] for r in results])),
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_speed": float(np.mean([r["avg_speed"] for r in results])),
        "avg_auto_brakes": float(np.mean([r["auto_brakes"] for r in results])),
    }


def overall(summaries):
    """One combined line over all scenarios (episode-weighted)."""
    total = sum(s["episodes"] for s in summaries)
    if total == 0:
        return summarize("ALL", [])

    def weighted(key):
        return sum(s[key] * s["episodes"] for s in summaries) / total

    return {
        "scenario": "ALL", "episodes": total,
        "success_pct": weighted("success_pct"), "collision_pct": weighted("collision_pct"),
        "double_line_pct": weighted("double_line_pct"), "timeout_pct": weighted("timeout_pct"),
        "avg_reward": weighted("avg_reward"), "avg_steps": weighted("avg_steps"),
        "avg_speed": weighted("avg_speed"), "avg_auto_brakes": weighted("avg_auto_brakes"),
    }


def table_lines(summaries):
    """The results table as a list of text lines (used for the console and the on-screen report)."""
    header = "{:<14} {:>4} {:>8} {:>10} {:>10} {:>9} {:>10} {:>8} {:>8}".format(
        "Scenario", "Eps", "Success%", "Collision%", "LineCross%", "Timeout%", "AvgReward", "AvgSteps", "AvgSpeed")
    lines = [header, "-" * len(header)]
    rows = list(summaries)
    if len(rows) > 1:
        rows.append(overall(summaries))
    for s in rows:
        if s["scenario"] == "ALL":
            lines.append("-" * len(header))
        lines.append("{:<14} {:>4d} {:>8.1f} {:>10.1f} {:>10.1f} {:>9.1f} {:>10.1f} {:>8.0f} {:>8.2f}".format(
            s["scenario"], s["episodes"], s["success_pct"], s["collision_pct"], s["double_line_pct"],
            s["timeout_pct"], s["avg_reward"], s["avg_steps"], s["avg_speed"]))
    return lines


def print_table(summaries, title="TEST RESULTS"):
    print("")
    print(title)
    for line in table_lines(summaries):
        print(line)


def save_results(summaries, csv_path, json_path, model_description, episodes, seed):
    """Write the summaries to a CSV and a JSON file (folders are created as needed)."""
    for path in (csv_path, json_path):
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    rows = list(summaries)
    if len(rows) > 1:
        rows.append(overall(summaries))

    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for s in rows:
            writer.writerow({
                key: (round(s[key], 3) if isinstance(s[key], float) else s[key]) for key in CSV_FIELDS
            })

    with open(json_path, "w") as handle:
        json.dump({
            "model": model_description,
            "test_episodes_per_scenario": episodes,
            "test_seed": seed,
            "results": rows,
        }, handle, indent=2)
    print("Saved {} and {}".format(csv_path, json_path))