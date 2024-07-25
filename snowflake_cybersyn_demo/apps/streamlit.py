from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional

import pandas as pd
import streamlit as st
from llama_agents import LlamaAgentsClient
from llama_index.core.llms import ChatMessage, ChatResponseGen
from llama_index.llms.openai import OpenAI


class TaskStatus(str, Enum):
    HUMAN_REQUIRED = "human_required"
    COMPLETED = "completed"
    SUBMITTED = "submitted"


@dataclass
class TaskModel:
    task_id: str
    input: str
    status: TaskStatus
    prompt: Optional[str] = None
    chat_history: List[ChatMessage] = field(default_factory=list)


def _llama_index_stream_wrapper(
    llama_index_stream: ChatResponseGen,
) -> Generator[str, Any, Any]:
    for chunk in llama_index_stream:
        yield chunk.delta


def _handle_task_submission(llama_agents_client: LlamaAgentsClient) -> None:
    """Handle the user submitted message. Clear task submission box, and
    add the new task to the submitted list.
    """

    # create new task and store in state
    task_input = st.session_state.task_input
    task_id = llama_agents_client.create_task(task_input)
    task = TaskModel(
        task_id=task_id,
        input=task_input,
        chat_history=[
            message,
            ChatMessage(
                role="assistant",
                content=f"Successfully submitted task: {task_id}.",
            ),
        ],
        status=TaskStatus.SUBMITTED,
    )
    st.session_state.submitted_pills.append(st.session_state.task_input)
    st.session_state.tasks.append(task)


llm = OpenAI(model="gpt-4o-mini")
control_plane_host = "0.0.0.0"
control_plane_port = 8001
llama_agents_client = LlamaAgentsClient(
    control_plane_url=(
        f"http://{control_plane_host}:{control_plane_port}"
        if control_plane_port
        else f"http://{control_plane_host}"
    )
)
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
        on_change=_handle_task_submission,
        args=(llama_agents_client,),
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
            response = st.write_stream(_llama_index_stream_wrapper(stream))
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
