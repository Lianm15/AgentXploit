from database import get_connection


class AttackMemory:
    """
    Reads cross-session technique performance from technique_history to:
      1. Seed AttackStrategyController UCB1 priors at the start of each session.
      2. Power the Intelligence dashboard with aggregate effectiveness data.

    No writes — technique_history is already populated by AttackStrategyController
    via save_technique_record() on every attempt. This class is read-only.
    """

    def load_priors(self, target_model: str) -> dict[str, float]:
        """
        Returns {technique: avg_compliance} for techniques with >= 2 recorded uses
        across all past sessions against the same target model.

        The HAVING COUNT(*) >= 2 guard filters single-attempt noise so that one
        lucky or unlucky result does not distort the prior.
        """
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT th.technique,
                   ROUND(AVG(th.compliance_score), 4) AS avg_compliance
            FROM technique_history th
            JOIN sessions s ON th.session_id = s.session_id
            WHERE s.target_model = ?
            GROUP BY th.technique
            HAVING COUNT(*) >= 2
            """,
            (target_model,),
        )
        rows = cursor.fetchall()
        conn.close()
        return {row["technique"]: row["avg_compliance"] for row in rows}

    def get_technique_effectiveness(self) -> dict:
        """
        Returns per-model technique performance aggregated across all sessions.
        Shape: {target_model: {technique: {avg_compliance, total_tries, sessions_used}}}
        Used by the Intelligence dashboard tab.
        """
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                s.target_model,
                th.technique,
                ROUND(AVG(th.compliance_score), 4) AS avg_compliance,
                COUNT(*)                            AS total_tries,
                COUNT(DISTINCT th.session_id)       AS sessions_used
            FROM technique_history th
            JOIN sessions s ON th.session_id = s.session_id
            GROUP BY s.target_model, th.technique
            ORDER BY s.target_model, avg_compliance DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()

        result: dict = {}
        for row in rows:
            model = row["target_model"]
            if model not in result:
                result[model] = {}
            result[model][row["technique"]] = {
                "avg_compliance": row["avg_compliance"],
                "total_tries": row["total_tries"],
                "sessions_used": row["sessions_used"],
            }
        return result

    def get_failure_distribution(self) -> dict:
        """
        Counts each failure_type across all recorded messages.
        Shape: {failure_type: count}
        Used by the Intelligence dashboard tab.
        """
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT failure_type, COUNT(*) AS count
            FROM messages
            WHERE failure_type IS NOT NULL
            GROUP BY failure_type
            ORDER BY count DESC
            """
        )
        rows = cursor.fetchall()
        conn.close()
        return {row["failure_type"]: row["count"] for row in rows}
