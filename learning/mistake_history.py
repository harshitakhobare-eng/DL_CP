"""Persistent Mistake History and Learning Storage.

Uses SQLite database to record student solution attempts, mistake classifications,
and practice records. Provides clean data access functions.
"""

from __future__ import annotations
from datetime import datetime, date
import json
import logging
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/learning_system.db"


class MistakeHistoryDB:
    """Data Access Object for student attempts, mistakes, and practice."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        """Initialize database tables with indexes."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Mistake History Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mistake_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    problem TEXT NOT NULL,
                    recognized_expression TEXT,
                    student_solution TEXT,
                    student_step TEXT,
                    correct_step TEXT,
                    mistake_type TEXT NOT NULL,
                    step_number INTEGER,
                    explanation TEXT,
                    recognition_confidence REAL,
                    final_correct INTEGER NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mistake_topic ON mistake_history(topic)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mistake_type ON mistake_history(mistake_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mistake_timestamp ON mistake_history(timestamp)")

            # Practice Records Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS practice_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    problem_latex TEXT NOT NULL,
                    solution_latex TEXT NOT NULL,
                    student_answer TEXT,
                    is_correct INTEGER NOT NULL,
                    mistake_type TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_practice_timestamp ON practice_records(timestamp)")
            conn.commit()

    def record_attempt(
        self,
        problem: str,
        topic: str,
        student_solution: List[str],
        final_correct: bool,
        recognized_expression: Optional[str] = None,
        mistake_type: Optional[str] = None,
        step_number: Optional[int] = None,
        student_step: Optional[str] = None,
        correct_step: Optional[str] = None,
        explanation: Optional[str] = None,
        recognition_confidence: float = 1.0,
    ) -> int:
        """Insert an attempt or detected mistake into the database."""
        now = datetime.now().isoformat()
        sol_json = json.dumps(student_solution)
        m_type = mistake_type if mistake_type else ("Correct" if final_correct else "Unknown/Other")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO mistake_history (
                    timestamp, topic, problem, recognized_expression,
                    student_solution, student_step, correct_step,
                    mistake_type, step_number, explanation,
                    recognition_confidence, final_correct
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now, topic, problem, recognized_expression or problem,
                sol_json, student_step or "", correct_step or "",
                m_type, step_number or 0, explanation or "",
                recognition_confidence, 1 if final_correct else 0
            ))
            conn.commit()
            return cursor.lastrowid

    def record_practice_result(
        self,
        topic: str,
        problem_latex: str,
        solution_latex: str,
        student_answer: str,
        is_correct: bool,
        mistake_type: Optional[str] = None,
    ) -> int:
        """Record the result of a personalized practice attempt."""
        now = datetime.now().isoformat()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO practice_records (
                    timestamp, topic, problem_latex, solution_latex,
                    student_answer, is_correct, mistake_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                now, topic, problem_latex, solution_latex,
                student_answer, 1 if is_correct else 0, mistake_type or ""
            ))
            conn.commit()
            return cursor.lastrowid

    def get_mistakes(
        self,
        limit: int = 50,
        topic: Optional[str] = None,
        mistake_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored mistakes."""
        query = "SELECT * FROM mistake_history WHERE final_correct = 0"
        params: List[Any] = []

        if topic:
            query += " AND topic = ?"
            params.append(topic)
        if mistake_type:
            query += " AND mistake_type = ?"
            params.append(mistake_type)

        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    def get_all_attempts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all attempts (correct and incorrect)."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM mistake_history ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_mistake_counts_by_type(self) -> Dict[str, int]:
        """Aggregate mistakes by taxonomy category."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT mistake_type, COUNT(*) as count
                FROM mistake_history
                WHERE final_correct = 0 AND mistake_type != 'Correct'
                GROUP BY mistake_type
                ORDER BY count DESC
            """).fetchall()
            return {r["mistake_type"]: r["count"] for r in rows}

    def get_topic_statistics(self) -> List[Dict[str, Any]]:
        """Compute attempts, correct count, and accuracy per topic."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT 
                    topic,
                    COUNT(*) as total_attempts,
                    SUM(final_correct) as total_correct
                FROM mistake_history
                GROUP BY topic
            """).fetchall()

            result = []
            for r in rows:
                total = r["total_attempts"]
                correct = r["total_correct"] or 0
                acc = round((correct / total) * 100, 1) if total > 0 else 0.0
                result.append({
                    "topic": r["topic"],
                    "total_attempts": total,
                    "total_correct": correct,
                    "accuracy": acc,
                })
            return result

    def get_learning_streak(self) -> int:
        """Calculate continuous day streak of learning activity."""
        with self._get_connection() as conn:
            rows = conn.execute("""
                SELECT DISTINCT SUBSTR(timestamp, 1, 10) as day
                FROM mistake_history
                UNION
                SELECT DISTINCT SUBSTR(timestamp, 1, 10) as day
                FROM practice_records
                ORDER BY day DESC
            """).fetchall()

            if not rows:
                return 0

            days = [datetime.strptime(r["day"], "%Y-%m-%d").date() for r in rows]
            today = date.today()
            
            # Check if active today or yesterday
            if days[0] < today and (today - days[0]).days > 1:
                return 0

            streak = 1
            for i in range(len(days) - 1):
                if (days[i] - days[i + 1]).days == 1:
                    streak += 1
                else:
                    break
            return streak

    def clear(self) -> None:
        """Clear all stored data (used in tests)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM mistake_history")
            conn.execute("DELETE FROM practice_records")
            conn.commit()
