import asyncio
import logging
import queue
import time
import threading
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.apps.async_list import AsyncSafeList
from snowflake_cybersyn_demo.apps.controller import (
    Controller,
    TaskModel,
    TaskResult,
    TaskStatus,
)
from snowflake_cybersyn_demo.apps.final_task_consumer import FinalTaskConsumer

logger = logging.getLogger(__name__)

llm = OpenAI(model="gpt-4o-mini")
control_plane_host = "0.0.0.0"
control_plane_port = 8001
human_input_request_queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
human_input_result_queue: asyncio.Queue[str] = asyncio.Queue()


st.set_page_config(layout="wide")


@st.cache_resource
def startup():
    human_input_request_queue: asyncio.Queue[Dict[str, str]] = asyncio.Queue()
    human_input_result_queue: asyncio.Queue[str] = asyncio.Queue()
    submitted_tasks_queue = queue.Queue()
    completed_tasks_queue = queue.Queue()
    controller = Controller(
        human_in_loop_queue=human_input_request_queue,
        human_in_loop_result_queue=human_input_result_queue,
        control_plane_host=control_plane_host,
        control_plane_port=control_plane_port,
        submitted_tasks_queue=submitted_tasks_queue,
    )

    final_task_consumer = FinalTaskConsumer(
        message_queue=controller._human_service.message_queue,
        completed_tasks_queue=completed_tasks_queue,
    )

    async def start_consuming_finalized_tasks(final_task_consumer):
        final_task_consuming_callable = (
            await final_task_consumer.register_to_message_queue()
        )

        await final_task_consuming_callable()

    # server thread will remain active as long as streamlit thread is running, or is manually shutdown
    thread = threading.Thread(
        name="Consuming thread",
        target=asyncio.run,
        args=(start_consuming_finalized_tasks(final_task_consumer),),
        daemon=False,
    )
    thread.start()

    time.sleep(5)
    st.session_state.consuming = True
    logger.info("Started consuming.")

    return controller, submitted_tasks_queue, completed_tasks_queue, final_task_consumer


controller, submitted_tasks_queue, completed_tasks_queue, final_task_consumer = (
    startup()
)


### App
st.title("Human In The Loop W/ LlamaAgents")

# state management
if "submitted_tasks" not in st.session_state:
    st.session_state["submitted_tasks"] = []
if "human_required_tasks" not in st.session_state:
    st.session_state["human_required_tasks"] = []
if "completed_tasks" not in st.session_state:
    st.session_state["completed_tasks"] = []
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []
if "consuming" not in st.session_state:
    st.session_state.consuming = False


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
            response = st.write_stream(controller._llama_index_stream_wrapper(stream))
        st.session_state.messages.append({"role": "assistant", "content": response})


@st.experimental_fragment(run_every="30s")
def task_df():
    st.text("Task Status")
    st.button("Refresh")
    tasks = (
        [t.input for t in st.session_state.submitted_tasks]
        + [t.input for t in st.session_state.human_required_tasks]
        + [t.input for t in st.session_state.completed_tasks]
    )

    status = (
        ["submitted"] * len(st.session_state.submitted_tasks)
        + ["human_required"] * len(st.session_state.human_required_tasks)
        + ["completed"] * len(st.session_state.completed_tasks)
    )
    data = {"tasks": tasks, "status": status}
    logger.info(f"data: {data}")
    df = pd.DataFrame(data)
    st.dataframe(
        df, selection_mode="single-row", use_container_width=True
    )  # Same as st.write(df)


task_df()


@st.experimental_fragment(run_every=5)
def process_completed_tasks(completed_queue: queue.Queue):
    task_res: Optional[TaskResult] = None
    try:
        task_res = completed_queue.get_nowait()
        logger.info("got new task result")
    except queue.Empty:
        logger.info(f"task result queue is empty.")

    if task_res:
        try:
            task_list = st.session_state.get("submitted_tasks")
            print(f"submitted tasks: {task_list}")
            ix, task = next(
                (ix, t)
                for ix, t in enumerate(task_list)
                if t.task_id == task_res.task_id
            )
            task.status = TaskStatus.COMPLETED
            task.chat_history.append(
                ChatMessage(role="assistant", content=task_res.result)
            )
            del task_list[ix]
            st.session_state.submitted_tasks = task_list
            st.session_state.completed_tasks.append(task)
            logger.info("updated submitted and completed tasks list.")
        except StopIteration:
            raise ValueError("Cannot find task in list of tasks.")


process_completed_tasks(completed_queue=completed_tasks_queue)
