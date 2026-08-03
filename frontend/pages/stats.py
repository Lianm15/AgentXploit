"""Streamlit multipage file - Streamlit auto-serves anything under pages/ at
a route matching its filename, so this becomes the /stats page automatically."""

import streamlit as st
import plotly.graph_objects as go
from api_client import ApiClient


def load_css(file):
    """Inject a local CSS file into the page - Streamlit has no native stylesheet linking."""
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def fmt(value, suffix=""):
    """Format a possibly-missing stat for display: None -> em dash, else value+suffix."""
    if value is None:
        return "—"
    return f"{value}{suffix}"


# ── Plotly chart helpers ──────────────────────────────────────────────────────
# Shared layout fragments so every chart on this page has the same transparent
# background / muted axis look without repeating the style dict per chart.

_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", size=12),
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.07)",
    zeroline=False,
    color="#64748b",
    title=None,
)

_NO_BAR = {"displayModeBar": False}

# Session status -> bar color for status_hbar_chart's per-bar palette.
_STATUS_PALETTE = {
    "success_found": "#10b981",
    "finished":      "#6366f1",
    "running":       "#22c55e",
    "paused":        "#f59e0b",
    "failed":        "#ef4444",
    "initialized":   "#64748b",
}


def donut_chart(success: int, failed: int, total: int) -> go.Figure:
    """Success/failure ratio donut with the session total centered inside the hole."""
    fig = go.Figure(go.Pie(
        labels=["Successful", "Failed"],
        values=[success, failed],
        hole=0.68,
        marker=dict(
            colors=["#6366f1", "#ef4444"],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textinfo="none",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b style='font-size:22px;color:#f1f5f9'>{total}</b>"
             f"<br><span style='font-size:11px;color:#64748b'>sessions</span>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#f1f5f9"),
    )
    fig.update_layout(
        **_CHART_BASE,
        height=220,
        showlegend=True,
        legend=dict(
            font=dict(color="#94a3b8", size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.08,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=0, r=0, t=8, b=28),
    )
    return fig


def hbar_chart(labels: list, values: list, color, value_suffix: str = "", height: int | None = None) -> go.Figure:
    """Horizontal bar chart used for per-model and per-status breakdowns.

    `color` accepts either a single color string (applied to every bar) or a
    list of per-bar colors — Plotly's marker.color takes both, which lets
    status_hbar_chart reuse this instead of duplicating the whole layout.
    """
    h = height or max(160, len(labels) * 44 + 40)
    text = [f"{v}{value_suffix}" for v in values]
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker=dict(color=color, opacity=0.88),
        text=text,
        textposition="outside",
        textfont=dict(color="#94a3b8", size=11),
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_BASE,
        height=h,
        margin=dict(l=4, r=32, t=16, b=4),
        xaxis=dict(**_AXIS_STYLE, tickfont=dict(size=10)),
        yaxis=dict(
            color="#94a3b8",
            title=None,
            tickfont=dict(size=11),
            autorange="reversed",
        ),
    )
    return fig


def status_hbar_chart(statuses: list, counts: list) -> go.Figure:
    """Session-status count bar chart - color-codes each bar by its status via `_STATUS_PALETTE`."""
    colors = [_STATUS_PALETTE.get(s, "#94a3b8") for s in statuses]
    return hbar_chart(statuses, counts, colors)


# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(page_title="AgentXploit — Statistics", layout="wide")

# Same custom-chrome-only approach as app.py — see that file for why.
st.markdown("""
    <style>
    [data-testid="stHeader"], header  { display: none !important; }
    [data-testid="stSidebar"],
    [data-testid="stSidebarNav"]      { display: none !important; }
    [data-testid="stDeployButton"]    { display: none !important; }
    [data-testid="stToolbar"]         { display: none !important; }
    #MainMenu                         { display: none !important; }
    section[data-testid="stMain"] > div { padding-top: 0 !important; }
    </style>
""", unsafe_allow_html=True)

load_css("styles/stats.css")

nav_col, _ = st.columns([2, 8])
with nav_col:
    st.page_link("app.py", label="← Home")

st.markdown('<h1 class="page-title">Statistics</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">Attack effectiveness across all completed sessions</p>',
    unsafe_allow_html=True,
)

client = ApiClient()

try:
    data = client.get_stats()
except Exception:
    # Fatal for this page — there's nothing to show without the stats payload.
    st.error("Backend not reachable.")
    st.stop()

if data["total_sessions"] == 0:
    st.info("No completed attack sessions yet. Run your first test to see statistics here.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab_overview, tab_models, tab_sessions, tab_intel = st.tabs(["Overview", "Models", "Sessions", "Intelligence"])

# ── Overview ──────────────────────────────────────────────────────────────────

with tab_overview:

    # Hero metrics — primary visual focus
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Sessions",     data["total_sessions"])
    c2.metric("Successful Attacks", data["successful_attacks"])
    c3.metric("Failed Attacks",     data["failed_attacks"])
    c4.metric("Success Rate",       fmt(data["success_rate"], "%"))

    if data["error_sessions"] > 0:
        st.write("")
        st.markdown('<p class="section-label">Operational</p>', unsafe_allow_html=True)
        err_col, info_col = st.columns([1, 3])
        err_col.metric("Crashed Sessions", data["error_sessions"])
        info_col.markdown(
            '<p style="font-size:0.8rem;color:#64748b;margin-top:1rem;">'
            "Sessions that ended due to a system error (e.g. Gemini API failure, "
            "Ollama timeout). Not counted as failed attacks — the attack loop never "
            "completed for these sessions."
            "</p>",
            unsafe_allow_html=True,
        )

    st.write("")

    # Donut chart + secondary metrics
    col_donut, col_secondary = st.columns([2, 3])

    with col_donut:
        st.markdown('<p class="section-label">Success vs Failed</p>', unsafe_allow_html=True)
        st.plotly_chart(
            donut_chart(data["successful_attacks"], data["failed_attacks"], data["total_sessions"]),
            use_container_width=True,
            config=_NO_BAR,
        )

    with col_secondary:
        st.markdown('<p class="section-label">Timing &amp; Attempts</p>', unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Duration",   fmt(data["avg_duration_seconds"], "s"))
        m2.metric("Fastest Win",    fmt(data["fastest_success_seconds"], "s"))
        m3.metric("Longest Attack", fmt(data["longest_attack_seconds"], "s"))
        st.write("")
        m4, m5 = st.columns(2)
        m4.metric("Avg Attempts / Session",   fmt(data["avg_attempts_per_session"]))
        m5.metric("Avg Attempts Until Win",   fmt(data["avg_attempts_until_success"]))
        st.write("")
        m6, m7 = st.columns(2)
        m6.metric("Total Prompts Generated",  data["total_prompts_generated"])
        m7.metric("Total Target Responses",   data["total_target_responses"])

# ── Models ────────────────────────────────────────────────────────────────────

with tab_models:

    if not data["per_model"]:
        st.info("No per-model data available yet.")
    else:
        models  = [r["model"]        for r in data["per_model"]]
        attacks = [r["attacks"]      for r in data["per_model"]]
        rates   = [r["success_rate"] for r in data["per_model"]]

        left, right = st.columns(2)

        with left:
            st.markdown('<p class="section-label">Attacks per Model</p>', unsafe_allow_html=True)
            st.plotly_chart(
                hbar_chart(models, attacks, "#6366f1"),
                use_container_width=True,
                config=_NO_BAR,
            )

        with right:
            st.markdown('<p class="section-label">Success Rate per Model</p>', unsafe_allow_html=True)
            st.plotly_chart(
                hbar_chart(models, rates, "#10b981", value_suffix="%"),
                use_container_width=True,
                config=_NO_BAR,
            )

        st.markdown('<p class="section-label">Breakdown</p>', unsafe_allow_html=True)
        st.dataframe(
            data["per_model"],
            column_config={
                "model":                "Model",
                "attacks":              "Attacks",
                "successes":            "Successes",
                "success_rate":         st.column_config.NumberColumn("Success Rate (%)", format="%.1f"),
                "avg_duration_seconds": st.column_config.NumberColumn("Avg Duration (s)", format="%.1f"),
            },
            use_container_width=True,
            hide_index=True,
        )

# ── Sessions ──────────────────────────────────────────────────────────────────

with tab_sessions:

    if data["status_distribution"]:
        statuses = [r["status"] for r in data["status_distribution"]]
        counts   = [r["count"]  for r in data["status_distribution"]]

        left, right = st.columns([3, 2])

        with left:
            st.markdown('<p class="section-label">Session Status Distribution</p>', unsafe_allow_html=True)
            st.plotly_chart(
                status_hbar_chart(statuses, counts),
                use_container_width=True,
                config=_NO_BAR,
            )

        with right:
            st.markdown('<p class="section-label">Counts</p>', unsafe_allow_html=True)
            st.dataframe(
                data["status_distribution"],
                column_config={"status": "Status", "count": "Count"},
                use_container_width=True,
                hide_index=True,
            )

    if data["recent_history"]:
        st.write("")
        st.markdown('<p class="section-label">Recent History (last 10)</p>', unsafe_allow_html=True)
        history = [
            {
                "Session":      row["session_id"][:8],
                "Model":        row["target_model"],
                "Duration (s)": round(row["time_elapsed"], 1),
                # target_count is the accurate attempt count; older rows lack it, so
                # fall back to messages_count // 3 (attacker + target + judge per attempt).
                "Attempts":     row.get("target_count", row["messages_count"] // 3),
                "Result":       "✓" if row["success"] in (1, True) else "✗",
            }
            for row in data["recent_history"]
        ]
        st.dataframe(
            history,
            column_config={
                "Result": st.column_config.TextColumn("Result", width="small"),
            },
            use_container_width=True,
            hide_index=True,
        )

# ── Intelligence ───────────────────────────────────────────────────────────────

with tab_intel:

    try:
        intel = client.get_intelligence_summary()
    except Exception:
        st.error("Could not load intelligence data from backend.")
        intel = None

    if intel is None:
        st.info("No intelligence data available yet.")
    else:
        effectiveness = intel.get("technique_effectiveness", {})
        failure_dist  = intel.get("failure_distribution", {})

        # ── Failure type distribution ──────────────────────────────────────────
        if failure_dist:
            st.markdown('<p class="section-label">Failure Type Distribution (all sessions)</p>',
                        unsafe_allow_html=True)
            st.plotly_chart(
                hbar_chart(
                    list(failure_dist.keys()),
                    list(failure_dist.values()),
                    "#6366f1",
                ),
                use_container_width=True,
                config=_NO_BAR,
            )
        else:
            st.info("No failure type data yet. Run at least one attack session.")

        # ── Technique effectiveness per model ──────────────────────────────────
        if effectiveness:
            st.write("")
            st.markdown('<p class="section-label">Technique Effectiveness per Model</p>',
                        unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.8rem;color:#64748b;margin-bottom:0.5rem;">'
                "These priors are loaded at the start of each new session against the same model "
                "to seed the UCB1 bandit's exploration order — techniques with higher historical "
                "compliance are tried first."
                "</p>",
                unsafe_allow_html=True,
            )

            for model, tech_data in sorted(effectiveness.items()):
                with st.expander(f"Model: {model}", expanded=True):
                    rows = [
                        {
                            "Technique":       tech,
                            "Avg Compliance":  round(stats["avg_compliance"], 3),
                            "Total Tries":     stats["total_tries"],
                            "Sessions Used":   stats["sessions_used"],
                        }
                        for tech, stats in sorted(
                            tech_data.items(),
                            key=lambda kv: kv[1]["avg_compliance"],
                            reverse=True,
                        )
                    ]
                    best = rows[0]["Technique"] if rows else "—"
                    st.caption(f"Best technique so far: **{best}** "
                               f"(avg compliance {rows[0]['Avg Compliance']:.3f})" if rows else "")
                    st.dataframe(
                        rows,
                        column_config={
                            "Avg Compliance": st.column_config.ProgressColumn(
                                "Avg Compliance",
                                min_value=0.0,
                                max_value=1.0,
                                format="%.3f",
                            ),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
        else:
            st.info("No technique effectiveness data yet. Run at least one attack session.")
