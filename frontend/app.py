"""Research Paper Agent — Streamlit frontend.

One paper at a time. Four tabs:
  1. Preview   — view the uploaded PDF in the browser
  2. Analyze   — kick off the agent pipeline, watch progress, see insights
  3. Chat      — multi-turn chat with tools (sections, citations, figures)
  4. Q&A       — single-shot Q&A (cheaper, no chat memory)
"""
from __future__ import annotations

import base64
import time
from typing import Any

import streamlit as st

from api_client import (
    APIError,
    ask_question,
    get_conversation,
    get_results,
    health,
    run_agent,
    send_message,
    start_conversation,
    upload_paper,
)
from components import (
    render_chat_history,
    render_insight,
    render_task_progress,
)

st.set_page_config(
    page_title="Research Paper Agent",
    page_icon="📄",
    layout="wide",
)


def _init_state() -> None:
    defaults = {
        "paper_id":        None,
        "paper_title":     None,
        "conversation_id": None,
        "agent_started":   False,
        "pdf_bytes":       None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


with st.sidebar:
    st.title("Paper")

    try:
        h = health()
    except APIError as e:
        st.error(f"Backend unreachable: {e.body}")
    except Exception as e:
        st.error(f"Backend error: {e}")

    st.divider()

    if st.session_state.paper_id:
        if st.button("Clear / upload new"):
            st.session_state.paper_id = None
            st.session_state.paper_title = None
            st.session_state.conversation_id = None
            st.session_state.agent_started = False
            st.session_state.pdf_bytes = None
            st.rerun()
    else:
        st.markdown("Upload a PDF to get started.")
        uploaded = st.file_uploader("PDF file", type=["pdf"], label_visibility="collapsed")
        if uploaded is not None:
            with st.spinner("Uploading and parsing PDF (figures take a moment)…"):
                try:
                    pdf_bytes = uploaded.getvalue()
                    res = upload_paper(uploaded.name, pdf_bytes)
                    st.session_state.paper_id = res["paper_id"]
                    st.session_state.paper_title = res["title"]
                    st.session_state.pdf_bytes = pdf_bytes
                    st.success(f"Uploaded: {res['title']}")
                    st.rerun()
                except APIError as e:
                    st.error(f"Upload failed: {e.body}")


if not st.session_state.paper_id:
    st.title("Research Paper Agent")
    st.info("Upload a paper from the sidebar to begin.")
    st.stop()

tab_preview, tab_analyze, tab_chat, tab_qa = st.tabs(
    ["Preview", "Analyze", "Chat", "Q&A"]
)


with tab_preview:
    if st.session_state.pdf_bytes:
        b64 = base64.b64encode(st.session_state.pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="900" style="border:1px solid #ddd;"></iframe>',
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "PDF preview is only available for the paper uploaded in this "
            "session. Click **Clear / upload new** in the sidebar and re-upload "
            "to preview."
        )


with tab_analyze:
    st.markdown(
        "Kick off the agent to analyze the paper. The agent will read the paper, extract key info, and generate an insight."
    )

    col1, col2 = st.columns([1, 3])
    with col1:
        goal = st.text_input(
            "Goal (optional)",
            placeholder="e.g. focus on weaknesses",
        )
        if st.button("Run agent", type="primary", use_container_width=True):
            try:
                run_agent(st.session_state.paper_id, goal)
                st.session_state.agent_started = True
                st.rerun()
            except APIError as e:
                st.error(f"Failed to start: {e.body}")

    with col2:
        if not st.session_state.agent_started:
            st.info("Click **Run agent** to start the analysis pipeline.")
        else:
            # Poll /results until all tasks are COMPLETED or FAILED.
            # st.empty() lets us update in place during the polling loop.
            placeholder = st.empty()
            for _ in range(60):  # cap at 60 polls = ~2 min
                try:
                    res = get_results(st.session_state.paper_id)
                except APIError as e:
                    placeholder.error(f"Results fetch failed: {e.body}")
                    break

                with placeholder.container():
                    render_task_progress(res["tasks"])
                    st.divider()
                    render_insight(res.get("insight"))

                statuses = {t["status"] for t in res["tasks"]}
                if statuses and statuses.issubset({"COMPLETED", "FAILED"}):
                    break
                time.sleep(2)


with tab_chat:
    st.markdown(
        "Have a multi-turn conversation about the paper. The agent can call "
        "tools when it needs specifics. This can be more expensive than single-turn Q&A, but allows for more complex interactions and follow-ups."
    )

    if not st.session_state.conversation_id:
        if st.button("Start new conversation", type="primary"):
            try:
                conv = start_conversation(st.session_state.paper_id)
                st.session_state.conversation_id = conv["id"]
                st.rerun()
            except APIError as e:
                st.error(f"Failed: {e.body}")
    else:
        st.caption(f"Conversation: `{st.session_state.conversation_id}`")
        if st.button("End / new conversation"):
            st.session_state.conversation_id = None
            st.rerun()

        try:
            conv = get_conversation(st.session_state.conversation_id)
            render_chat_history(conv["messages"])
        except APIError as e:
            st.error(f"Couldn't load conversation: {e.body}")
            st.stop()

        user_input = st.chat_input("Ask a question about this paper…")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Thinking… (agent may call tools)"):
                    try:
                        result = send_message(
                            st.session_state.conversation_id, user_input,
                        )
                    except APIError as e:
                        st.error(f"Agent failed: {e.body}")
                        st.stop()

                # Render the new trace (tool calls + final answer)
                for m in result["trace"]:
                    if m["role"] == "user":
                        continue   # already rendered above
                    if m["role"] == "assistant" and m.get("tool_name"):
                        with st.expander(f"🛠️ Used tool: `{m['tool_name']}`"):
                            st.json(m.get("tool_args") or {})
                    elif m["role"] == "tool":
                        with st.expander(f"📤 Result from `{m.get('tool_name', '?')}`"):
                            st.code(m["content"][:2000], language="text")
                    elif m["role"] == "assistant":
                        st.write(m["content"])

                st.caption(f"({result['iterations']} iteration(s))")


with tab_qa:
    st.markdown(
        "Ask one question. Cheaper and faster than chat and use very limited context."
    )

    question = st.text_area("Your question", height=100)
    if st.button("Ask", type="primary"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            with st.spinner("Asking the LLM…"):
                try:
                    res: dict[str, Any] = ask_question(
                        st.session_state.paper_id, question,
                    )
                    answer = res.get("answer")
                    if isinstance(answer, dict):
                        st.write(answer.get("answer", answer))
                    else:
                        st.write(answer)
                except APIError as e:
                    st.error(f"Question failed: {e.body}")
