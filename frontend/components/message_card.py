"""HTML template for a single chat message card."""

# Message dict schema (enforced by callers):
#   sender:      "agent" | "model"
#   time:        str
#   text:        str
#   attack_type: str | None


def render(msg: dict, model_name: str) -> str:
    """Return the HTML string for one message card."""
    is_agent   = msg["sender"] == "agent"
    avatar_cls = "avatar-ax" if is_agent else "avatar-ai"
    initials   = "AX"        if is_agent else "AI"
    name       = "AgentXploit" if is_agent else model_name

    badge = (
        f'<span class="attack-badge">{msg["attack_type"]}</span>'
        if msg.get("attack_type")
        else ""
    )

    return (
        '<div class="msg-card">'
        '  <div class="msg-header">'
        f'    <div class="avatar {avatar_cls}">{initials}</div>'
        '    <div class="msg-meta">'
        f'      <span class="msg-name">{name}</span>'
        f'      <span class="msg-time">{msg["time"]}</span>'
        "    </div>"
        "  </div>"
        f'  <div class="msg-body">{msg["text"]}</div>'
        f"  {badge}"
        "</div>"
    )
