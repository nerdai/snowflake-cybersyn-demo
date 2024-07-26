import asyncio
import logging
from typing import Dict, List
from contextvars import ContextVar

import pandas as pd
import streamlit as st
from llama_index.core.llms import ChatMessage
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.apps.controller import (
    Controller,
    TaskModel,
    TaskResult,
    TaskStatus,
)
from snowflake_cybersyn_demo.apps.async_list import AsyncSafeList

logger = logging.getLogger(__name__)

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
if "submitted_tasks" not in st.session_state:
    st.session_state["submitted_tasks"] = []
if "human_required_tasks" not in st.session_state:
    st.session_state["human_required_tasks"] = []
if "completed_tasks" not in st.session_state:
    st.session_state["completed_tasks"] = []
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
            response = st.write_stream(controller._llama_index_stream_wrapper(stream))
        st.session_state.messages.append({"role": "assistant", "content": response})


@st.cache_resource
def get_consuming_callables() -> None:
    async def launch() -> None:
        start_consuming_callable = (
            await controller._human_service.message_queue.register_consumer(
                controller._human_service.as_consumer()
            )
        )

        final_task_consuming_callable = (
            await controller._human_service.message_queue.register_consumer(
                controller._final_task_consumer
            )
        )

        return start_consuming_callable, final_task_consuming_callable

    return asyncio.run(launch())


start_consuming_callable, final_task_consuming_callable = get_consuming_callables()


def remove_from_list_closure(
    task_list: List[TaskModel],
    task_status: TaskStatus,
    task_res: TaskResult,
    # current_task: Tuple[int, TaskStatus] = current_task,
) -> None:
    """Closure depending on the task list/status.

    Returns a function used to move the task from the incumbent list/status
    over to the completed list.
    """
    ix, task = next(
        (ix, t) for ix, t in enumerate(task_list) if t.task_id == task_res.task_id
    )
    task.status = TaskStatus.COMPLETED
    task.chat_history.append(ChatMessage(role="assistant", content=task_res.result))
    del task_list[ix]

    # if current_task:
    #     current_task_ix, current_task_status = current_task
    #     if current_task_status == task_status and current_task_ix == ix:
    #         # current task is the task that is being moved to completed
    #         current_task = (len(completed) - 1, TaskStatus.COMPLETED)


stuff_lock = asyncio.Lock()


@st.cache_resource
def get_async_safe_lists():
    submitted_tasks = AsyncSafeList()
    return submitted_tasks


submitted_tasks = get_async_safe_lists()


async def listening_to_queue(ctr) -> None:
    logger.info("🤖 LISTENING")
    h_task = asyncio.create_task(start_consuming_callable())  # noqa: F841
    f_task = asyncio.create_task(final_task_consuming_callable())  # noqa: F841

    human_required_tasks = []
    completed_tasks = []
    while True:
        logger.info(f"submitted: {submitted_tasks._list}")
        # logger.info(f"completed: {completed_tasks}")

        try:
            new_task: TaskModel = controller._submitted_tasks_queue.get_nowait()
            await submitted_tasks.append(new_task)
            logger.info("got new submitted task")
            logger.info(f"submitted: {submitted_tasks}")
        except asyncio.QueueEmpty:
            logger.info("task completion queue is empty")

        # try:
        #     task_res: TaskResult = controller._completed_tasks_queue.get_nowait()
        #     logger.info("got new completed task result")
        # except asyncio.QueueEmpty:
        #     task_res = None
        #     logger.info("task completion queue is empty")

        # if task_res:
        #     if task_res.task_id in [t.task_id for t in submitted_tasks]:
        #         ix, task = next(
        #             (ix, t)
        #             for ix, t in enumerate(submitted_tasks)
        #             if t.task_id == task_res.task_id
        #         )
        #         task.status = TaskStatus.COMPLETED
        #         task.chat_history.append(
        #             ChatMessage(role="assistant", content=task_res.result)
        #         )
        #         del submitted_tasks[ix]
        #         completed_tasks.append(task)
        #         logger.info(f"updated task status from submitted to completed.")
        #     elif task_res.task_id in [t.task_id for t in human_required_tasks]:
        #         remove_from_list_closure(
        #             st.session_state.human_required_tasks,
        #             TaskStatus.HUMAN_REQUIRED,
        #         )

        ctr.text("Task Status")
        tasks = (
            [t.input for t in submitted_tasks]
            + [t.input for t in human_required_tasks]
            + [t.input for t in completed_tasks]
        )

        n_submitted = await submitted_tasks.length()
        status = (
            ["submitted"] * n_submitted
            + ["human_required"] * len(human_required_tasks)
            + ["completed"] * len(completed_tasks)
        )
        data = {"tasks": tasks, "status": status}
        logger.info(f"data: {data}")
        df = pd.DataFrame(data)
        ctr.dataframe(
            df, selection_mode="single-row", use_container_width=True
        )  # Same as st.write(df)
        await asyncio.sleep(5)


bottom = st.empty()

asyncio.run(listening_to_queue(bottom))
