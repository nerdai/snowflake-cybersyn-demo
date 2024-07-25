import asyncio
from typing import Dict

import pandas as pd
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.apps.controller import Controller

llm = OpenAI(model="gpt-4o-mini")
control_plane_host = "0.0.0.0"
control_plane_port = 8001
human_input_request_queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
human_input_result_queue: asyncio.Queue[str] = asyncio.Queue()
controller = Controller(
    human_in_loop_queue=human_input_request_queue,
    human_in_loop_result_queue=human_input_result_queue,
    control_plane_host=control_plane_host,
    control_plane_port=control_plane_port,
)

### App
st.set_page_config(layout="wide")
st.title("Human In The Loop W/ LlamaAgents")

# state management
if "submitted_pills" not in st.session_state:
    st.session_state["submitted_pills"] = []
st.session_state["human_required_pills"] = []
st.session_state["completed_pills"] = []
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []


left, right = st.columns([1, 2], vertical_alignment="bottom")

with left:
    task_input = st.text_input(
        "Task input",
        placeholder="Enter a task input.",
        key="task_input",
        on_change=controller._handle_task_submission,
    )

with right:
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
            response = st.write_stream(
                controller._llama_index_stream_wrapper(stream)
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

bottom = st.container()
with bottom:
    st.text("Task Status")
    tasks = (
        st.session_state.submitted_pills
        + st.session_state.human_required_pills
        + st.session_state.completed_pills
    )
    status = (
        ["submitted"] * len(st.session_state.submitted_pills)
        + ["human_required"] * len(st.session_state.human_required_pills)
        + ["completed"] * len(st.session_state.completed_pills)
    )
    data = {"tasks": tasks, "status": status}
    df = pd.DataFrame(data)
    st.dataframe(
        df, selection_mode="single-row", use_container_width=True
    )  # Same as st.write(df)
