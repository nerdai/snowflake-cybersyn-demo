import asyncio
import logging
import queue
import threading
import time
from typing import Optional

import pandas as pd
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.additional_services.human_in_the_loop import (
    HumanRequest,
    HumanService,
)
from snowflake_cybersyn_demo.apps.controller import (
    Controller,
    TaskResult,
    TaskStatus,
)
from snowflake_cybersyn_demo.apps.final_task_consumer import FinalTaskConsumer

logger = logging.getLogger(__name__)

llm = OpenAI(model="gpt-4o-mini")
control_plane_host = "0.0.0.0"
control_plane_port = 8001


st.set_page_config(layout="wide")


@st.cache_resource
def startup():
    from snowflake_cybersyn_demo.additional_services.human_in_the_loop import (
        human_input_request_queue,
        human_input_result_queue,
        human_service,
    )

    completed_tasks_queue = queue.Queue()
    controller = Controller(
        control_plane_host=control_plane_host,
        control_plane_port=control_plane_port,
    )

    async def start_consuming_human_tasks(human_service: HumanService):
        human_task_consuming_callable = (
            await human_service.message_queue.register_consumer(
                human_service.as_consumer()
            )
        )

        ht_task = asyncio.create_task(human_task_consuming_callable())

        launch_task = asyncio.create_task(human_service.processing_loop())

        await asyncio.Future()

    hr_thread = threading.Thread(
        name="Human Request thread",
        target=asyncio.run,
        args=(start_consuming_human_tasks(human_service),),
        daemon=False,
    )
    hr_thread.start()

    final_task_consumer = FinalTaskConsumer(
        message_queue=human_service.message_queue,
        completed_tasks_queue=completed_tasks_queue,
    )

    async def start_consuming_finalized_tasks(final_task_consumer):
        final_task_consuming_callable = (
            await final_task_consumer.register_to_message_queue()
        )

        await final_task_consuming_callable()

    # server thread will remain active as long as streamlit thread is running, or is manually shutdown
    ft_thread = threading.Thread(
        name="Consuming thread",
        target=asyncio.run,
        args=(start_consuming_finalized_tasks(final_task_consumer),),
        daemon=False,
    )
    ft_thread.start()

    time.sleep(5)
    logger.info("Started consuming.")

    return (
        controller,
        completed_tasks_queue,
        final_task_consumer,
        human_input_request_queue,
        human_input_result_queue,
    )


(
    controller,
    completed_tasks_queue,
    final_task_consumer,
    human_input_request_queue,
    human_input_result_queue,
) = startup()


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
            response = st.write_stream(
                controller._llama_index_stream_wrapper(stream)
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )


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
        logger.info("task result queue is empty.")

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


@st.experimental_fragment(run_every=5)
def process_human_input_requests(
    human_requests_queue: queue.Queue[HumanRequest],
):
    human_req: Optional[HumanRequest] = None
    try:
        human_req = human_requests_queue.get_nowait()
        logger.info("got new human request")
    except queue.Empty:
        logger.info("human request queue is empty.")

    if human_req:
        try:
            task_list = st.session_state.get("submitted_tasks")
            print(f"submitted tasks: {task_list}")
            ix, task = next(
                (ix, t)
                for ix, t in enumerate(task_list)
                if t.task_id == human_req["task_id"]
            )
            task.status = TaskStatus.COMPLETED
            task.chat_history.append(
                ChatMessage(role="assistant", content=human_req["prompt"])
            )
            del task_list[ix]
            st.session_state.submitted_tasks = task_list
            st.session_state.human_required_tasks.append(task)
            logger.info("updated submitted and human required tasks list.")
        except StopIteration:
            raise ValueError("Cannot find task in list of tasks.")


process_human_input_requests(human_requests_queue=human_input_request_queue)
