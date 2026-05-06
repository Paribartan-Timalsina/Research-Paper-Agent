"""Reusable Streamlit rendering helpers."""
from __future__ import annotations

from typing import Any

import streamlit as st

_STATUS_BADGES = {
    "PENDING":   ("⚪", "gray"),
    "RUNNING":   ("🔄", "blue"),
    "COMPLETED": ("✅", "green"),
    "FAILED":    ("❌", "red"),
}


def render_task_progress(tasks: list[dict[str, Any]]) -> None:
    """Render a vertical list of agent-task statuses."""
    if not tasks:
        st.info("No tasks scheduled yet — click **Run agent** to start.")
        return

    completed = sum(1 for t in tasks if t["status"] == "COMPLETED")
    total = len(tasks)
    st.progress(completed / total, text=f"{completed} / {total} steps complete")

    for t in tasks:
        icon, _ = _STATUS_BADGES.get(t["status"], ("•", "gray"))
        line = f"{icon}  **{t['task_name']}** — `{t['status']}`"
        if t.get("error"):
            line += f"  \n  ⚠️  {t['error']}"
        st.markdown(line)


def render_insight(insight: dict[str, Any] | None) -> None:
    """Render the consolidated Insight row, section by section as it fills in."""
    if not insight:
        st.info("No insight yet — results will appear here as steps complete.")
        return

    if insight.get("summary"):
        st.subheader("Summary")
        st.write(insight["summary"])

    if insight.get("contributions"):
        st.subheader("Contributions")
        for item in insight["contributions"]:
            st.markdown(f"- {item}")

    if insight.get("methodology"):
        st.subheader("Methodology")
        st.write(insight["methodology"])

    if insight.get("limitations"):
        st.subheader("Limitations")
        for item in insight["limitations"]:
            st.markdown(f"- {item}")

    if insight.get("future_work"):
        st.subheader("Future work")
        for item in insight["future_work"]:
            st.markdown(f"- {item}")


def render_chat_message(msg: dict[str, Any]) -> None:
    """Render a single message row from /conversations/{id}.

    msg = {role, content, tool_name, tool_args, ...}
    """
    role = msg["role"]
    if role == "user":
        with st.chat_message("user"):
            st.write(msg["content"])
    elif role == "assistant":
        if msg.get("tool_name"):
            # Tool-call turn — collapsible
            with st.chat_message("assistant"):
                with st.expander(f"🛠️ Used tool: `{msg['tool_name']}`"):
                    st.json(msg.get("tool_args") or {})
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    elif role == "tool":
        with st.chat_message("assistant", avatar="🧰"):
            with st.expander(f"📤 Result from `{msg.get('tool_name', '?')}`"):
                st.code(msg["content"][:2000], language="text")


def render_chat_history(messages: list[dict[str, Any]]) -> None:
    """Render a full conversation, oldest first."""
    for m in messages:
        render_chat_message(m)
