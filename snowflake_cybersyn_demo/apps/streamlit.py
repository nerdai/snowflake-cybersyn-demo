import asyncio
import logging
from typing import Dict, List

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

bottom = st.container()
with bottom:
    st.text("Task Status")
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
    df = pd.DataFrame(data)
    st.dataframe(
        df, selection_mode="single-row", use_container_width=True
    )  # Same as st.write(df)


# regularly check human input queue
@st.experimental_fragment(run_every=2)
def continuously_check_for_human_required() -> None:
    try:
        dict: Dict[str, str] = human_input_request_queue.get_nowait()
        prompt = dict.get("prompt")
        task_id = dict.get("task_id")
        logger.info(f"prompt: {prompt}, task_id: {task_id}")

        # find task with the provided task_id
        try:
            ix, task = next(
                (ix, t)
                for ix, t in enumerate(st.session_state.submitted_tasks)
                if t.task_id == task_id
            )
            task.prompt = prompt
            task.status = TaskStatus.HUMAN_REQUIRED
            task.chat_history += [
                ChatMessage(
                    role="assistant",
                    content="Human assistance is required.",
                ),
                ChatMessage(role="assistant", content=prompt),
            ]

            del st.session_state.submitted_tasks[ix]
            st.session_state.human_required_tasks.append(task)

            # if current_task:
            #     current_task_ix, current_task_status = current_task
            #     if (
            #         current_task_status == TaskStatus.SUBMITTED
            #         and current_task_ix == ix
            #     ):
            #         current_task = (
            #             len(st.session_state.human_required_tasks) - 1,
            #             TaskStatus.HUMAN_REQUIRED,
            #         )

        except StopIteration:
            raise ValueError("Cannot find task in list of tasks.")
        logger.info("appended human input request.")
    except asyncio.QueueEmpty:
        logger.info("human input request queue is empty.")
        pass


@st.experimental_fragment(run_every=5)
def continuously_check_for_completed_tasks() -> None:
    """Logic used when polling the completed tasks queue.

    Specifically, move tasks from either submitted/human-required status to
    completed status.
    """

    def remove_from_list_closure(
        task_list: List[TaskModel],
        task_status: TaskStatus,
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
        st.session_state.completed_tasks.append(task)

        # if current_task:
        #     current_task_ix, current_task_status = current_task
        #     if current_task_status == task_status and current_task_ix == ix:
        #         # current task is the task that is being moved to completed
        #         current_task = (len(completed) - 1, TaskStatus.COMPLETED)

    try:
        task_res: TaskResult = controller._completed_tasks_queue.get_nowait()
        logger.info("got new completed task result")
        if task_res.task_id in [t.task_id for t in st.session_state.submitted_tasks]:
            remove_from_list_closure(
                st.session_state.submitted_tasks, TaskStatus.SUBMITTED
            )
        elif task_res.task_id in [
            t.task_id for t in st.session_state.human_required_tasks
        ]:
            remove_from_list_closure(
                st.session_state.human_required_tasks,
                TaskStatus.HUMAN_REQUIRED,
            )
        else:
            raise ValueError("Completed task not in submitted or human_needed lists.")
    except asyncio.QueueEmpty:
        logger.info("completed task queue is empty.")
        pass


continuously_check_for_human_required()
continuously_check_for_completed_tasks()


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


async def listening_to_queue() -> None:
    h_task = asyncio.create_task(start_consuming_callable())  # noqa: F841
    f_task = asyncio.create_task(final_task_consuming_callable())  # noqa: F841
    while True:
        await asyncio.sleep(0.1)


asyncio.run(listening_to_queue())
