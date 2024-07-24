from typing import Any, Generator

import streamlit as st
from llama_index.core.llms import ChatMessage, ChatResponseGen
from llama_index.llms.openai import OpenAI
from streamlit_pills import pills


def _llama_index_stream_wrapper(
    llama_index_stream: ChatResponseGen,
) -> Generator[str, Any, Any]:
    for chunk in llama_index_stream:
        yield chunk.delta


def _handle_task_submission() -> None:
    st.session_state.submitted_pills.append(st.session_state.task_input)


llm = OpenAI(model="gpt-4o-mini")
st.set_page_config(layout="wide")
st.title("Human In The Loop W/ LlamaAgents")

# state management
if "submitted_pills" not in st.session_state:
    st.session_state["submitted_pills"] = []
st.session_state["human_required_pills"] = []
st.session_state["completed_pills"] = []


left, middle, right = st.columns([1, 2, 1], vertical_alignment="bottom")

with left:
    task_input = st.text_input(
        "Task input",
        placeholder="Enter a task input.",
        key="task_input",
        on_change=_handle_task_submission,
    )

with middle:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = llm.stream_chat(
                messages=[
                    ChatMessage(role=m["role"], content=m["content"])
                    for m in st.session_state.messages
                ]
            )
            response = st.write_stream(_llama_index_stream_wrapper(stream))
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

with right:
    if st.session_state.submitted_pills:
        submitted_pills = st.session_state.submitted_pills
        submitted = pills(
            "Submitted",
            options=submitted_pills,
            key="selected_submitted",
        )

    if st.session_state.human_required_pills:
        human_required = pills(
            "Human Required",
            st.session_state.human_required_pills,
            key="selected_human_required",
        )

    if st.session_state.completed_pills:
        completed = pills(
            "Completed",
            st.session_state.completed_pills,
            key="selected_completed",
        )
