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
    font=dict(family="Hanken Grotesk, sans-serif", color="#c5c9d3", size=12),
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor="rgba(255,255,255,0.07)",
    zeroline=False,
    color="#6b7180",
    title=None,
)

_NO_BAR = {"displayModeBar": False}

# Reused for charts with several distinct categorical bars (e.g. failure
# types) where a single flat color makes every bar equally hard to tell
# apart. Cycles through colors already used elsewhere in the app instead
# of inventing new ones.
_CATEGORY_CYCLE = ["#ffb020", "#6366f1", "#ff6b4a", "#8b5cf6", "#6b7180"]

# Session status -> bar color for status_hbar_chart's per-bar palette.
# Matches app.py's _STATUS_COLORS exactly — the live status pill is the
# canonical source for what each status "means" colorwise, so this chart
# should never invent its own mapping. success_found/running are genuinely
# green there too; finished is indigo, not amber.
_STATUS_PALETTE = {
    "success_found": "#22c55e",
    "finished":      "#6366f1",
    "running":       "#22c55e",
    "paused":        "#eab308",
    "failed":        "#ef4444",
    "initialized":   "#6b7180",
}

# Client-facing labels for raw DB status values. "failed" only ever means the
# attack loop itself crashed (system/API error) - never that the test simply
# didn't succeed - so it reads as "Not Finished" rather than the harsher/more
# alarming "Failed" or "Crashed".
_STATUS_LABELS = {
    "success_found": "Successful",
    "finished":      "Finished",
    "running":       "Running",
    "paused":        "Paused",
    "failed":        "Not Finished",
    "initialized":   "Initialized",
}

# Client-facing labels for classify_zone()'s raw zone_a/b/c/d/ambiguous labels
# (backend/strategy_controller.py) - short, jargon-free descriptions of what
# actually happened on that attempt instead of an opaque "Zone A/B/C/D" name.
_OUTCOME_LABELS = {
    "zone_a":    "Near Full Bypass",
    "zone_b":    "Bypassed, No Content",
    "zone_c":    "Fully Blocked",
    "zone_d":    "Leaked Despite Refusal",
    "ambiguous": "Inconclusive",
}


def donut_chart(success: int, failed: int, total: int) -> go.Figure:
    """Success/failure ratio donut with the session total centered inside the hole."""
    fig = go.Figure(go.Pie(
        labels=["Successful", "Failed"],
        values=[success, failed],
        hole=0.68,
        marker=dict(
            colors=["#ffb020", "#ef4444"],
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        textinfo="none",
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b style='font-size:22px;color:#f4f5f7'>{total}</b>"
             f"<br><span style='font-size:11px;color:#6b7180'>sessions</span>",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(color="#f4f5f7"),
    )
    fig.update_layout(
        **_CHART_BASE,
        # height and margin.b both grow by the same 30px so the plot area
        # (height - margin.t - margin.b) stays 184px — the donut itself is
        # pixel-identical to before. The extra 30px is pure blank space
        # below the donut, giving the legend room to sit clear of it.
        height=250,
        showlegend=True,
        legend=dict(
            font=dict(color="#c5c9d3", size=11),
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(l=0, r=0, t=8, b=58),
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
        textfont=dict(color="#c5c9d3", size=11),
        hovertemplate="%{y}: %{x}<extra></extra>",
    ))
    fig.update_layout(
        **_CHART_BASE,
        height=h,
        margin=dict(l=4, r=32, t=16, b=4),
        xaxis=dict(**_AXIS_STYLE, tickfont=dict(size=10)),
        yaxis=dict(
            color="#c5c9d3",
            title=None,
            tickfont=dict(size=11),
            autorange="reversed",
        ),
    )
    return fig


def status_hbar_chart(statuses: list, counts: list) -> go.Figure:
    """Session-status count bar chart - color-codes each bar by its raw status via
    `_STATUS_PALETTE`, but labels each bar with the client-facing name from `_STATUS_LABELS`."""
    colors = [_STATUS_PALETTE.get(s, "#6b7180") for s in statuses]
    labels = [_STATUS_LABELS.get(s, s.replace("_", " ").title()) for s in statuses]
    return hbar_chart(labels, counts, colors)


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

# Nav link: kept structurally identical to app.py's "Statistics" link —
# same .block-container/[data-testid="stPageLink"] rules duplicated
# verbatim in stats.css/home.css (see either file's "Shared top nav"
# comment) — so both pages share the same box math, not just similar
# values nudged into visual agreement.
st.page_link("app.py", label="← Home")

st.markdown('<h1 class="page-title">Attack <span>Statistics</span></h1>', unsafe_allow_html=True)
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
        err_col.metric("Not Finished", data["error_sessions"])
        info_col.markdown(
            '<p style="font-size:0.8rem;color:#6b7180;margin-top:1rem;">'
            "Sessions that couldn't finish because of a temporary system issue "
            "(e.g. a connection or API hiccup) — not because the attack was tried "
            "and came up unsuccessful. These aren't counted in the success rate."
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
                hbar_chart(models, attacks, "#ffb020"),
                use_container_width=True,
                config=_NO_BAR,
            )

        with right:
            st.markdown('<p class="section-label">Success Rate per Model</p>', unsafe_allow_html=True)
            st.plotly_chart(
                hbar_chart(models, rates, "#6366f1", value_suffix="%"),
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
            status_counts = [
                {
                    "status": _STATUS_LABELS.get(r["status"], r["status"].replace("_", " ").title()),
                    "count": r["count"],
                }
                for r in data["status_distribution"]
            ]
            st.dataframe(
                status_counts,
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

        # ── Attempt outcome distribution ────────────────────────────────────────
        if failure_dist:
            st.markdown('<p class="section-label">Attempt Outcome Breakdown (all sessions)</p>',
                        unsafe_allow_html=True)
            outcome_keys = list(failure_dist.keys())
            failure_labels = [
                _OUTCOME_LABELS.get(k, k.replace("_", " ").title()) for k in outcome_keys
            ]
            failure_colors = [
                _CATEGORY_CYCLE[i % len(_CATEGORY_CYCLE)] for i in range(len(failure_labels))
            ]
            st.plotly_chart(
                hbar_chart(
                    failure_labels,
                    list(failure_dist.values()),
                    failure_colors,
                ),
                use_container_width=True,
                config=_NO_BAR,
            )
        else:
            st.info("No attempt outcome data yet. Run at least one attack session.")

        # ── Technique effectiveness per model ──────────────────────────────────
        if effectiveness:
            st.write("")
            st.markdown('<p class="section-label">Technique Effectiveness per Model</p>',
                        unsafe_allow_html=True)
            st.markdown(
                '<p style="font-size:0.8rem;color:#6b7180;margin-bottom:0.5rem;">'
                "These results carry over into each new session against the same model — "
                "techniques that performed best historically are tried first."
                "</p>",
                unsafe_allow_html=True,
            )

            for model, tech_data in sorted(effectiveness.items()):
                with st.expander(f"Model: {model}", expanded=True):
                    rows = [
                        {
                            "Technique":       tech.replace("_", " ").title(),
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
