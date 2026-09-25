"""Progress Analytics and Dashboard Metric Aggregator.

Calculates:
- Total problems attempted and total correct
- Overall and recent accuracy (last 10 attempts)
- Mistakes breakdown by taxonomy category
- Topic-wise accuracy & breakdown
- Learning activity timeline / progress over time
- Current learning streak
- Recurring weakness recommendations
"""

from __future__ import annotations
from datetime import datetime
from typing import Any, Dict, List, Optional
from learning.mistake_history import MistakeHistoryDB


class ProgressTracker:
    """Calculates live analytics from real student history."""

    def __init__(self, db: MistakeHistoryDB):
        self.db = db

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Generate comprehensive metrics for dashboard display."""
        attempts = self.db.get_all_attempts(limit=500)
        total_attempts = len(attempts)
        total_correct = sum(1 for a in attempts if a["final_correct"] == 1)
        
        overall_accuracy = round((total_correct / total_attempts) * 100, 1) if total_attempts > 0 else 0.0

        # Recent accuracy: last 10 attempts
        recent_10 = attempts[:10]
        recent_correct = sum(1 for a in recent_10 if a["final_correct"] == 1)
        recent_accuracy = round((recent_correct / len(recent_10)) * 100, 1) if recent_10 else 0.0

        # Mistakes by Category
        mistakes_by_category = self.db.get_mistake_counts_by_type()

        # Topic-wise performance
        topic_stats = self.db.get_topic_statistics()

        # Learning Streak
        streak = self.db.get_learning_streak()

        # Top recurring mistakes
        recurring_mistakes = sorted(
            [{"type": k, "count": v} for k, v in mistakes_by_category.items()],
            key=lambda x: x["count"],
            reverse=True,
        )

        # Timeline / Activity history (grouped by date)
        timeline_dict: Dict[str, Dict[str, int]] = {}
        for a in reversed(attempts):
            day = a["timestamp"][:10]
            if day not in timeline_dict:
                timeline_dict[day] = {"attempts": 0, "correct": 0}
            timeline_dict[day]["attempts"] += 1
            if a["final_correct"] == 1:
                timeline_dict[day]["correct"] += 1

        timeline = [
            {
                "date": d,
                "attempts": data["attempts"],
                "correct": data["correct"],
                "accuracy": round((data["correct"] / data["attempts"]) * 100, 1) if data["attempts"] > 0 else 0.0,
            }
            for d, data in timeline_dict.items()
        ]

        # Recommendation based on recurring mistake
        top_weakness = recurring_mistakes[0]["type"] if recurring_mistakes else None
        recommendation = (
            f"Focus on practice with '{top_weakness}' to strengthen foundational algebraic transformations."
            if top_weakness
            else "Great work! Continue solving problems to build your learning profile."
        )

        return {
            "total_attempts": total_attempts,
            "total_correct": total_correct,
            "overall_accuracy": overall_accuracy,
            "recent_accuracy": recent_accuracy,
            "streak_days": streak,
            "mistakes_by_category": mistakes_by_category,
            "topic_performance": topic_stats,
            "recurring_mistakes": recurring_mistakes,
            "timeline": timeline,
            "recommendation": recommendation,
        }
