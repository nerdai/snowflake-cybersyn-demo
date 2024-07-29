import asyncio
import logging
import queue
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional

import streamlit as st
from llama_agents import (
    CallableMessageConsumer,
    LlamaAgentsClient,
    QueueMessage,
)
from llama_agents.types import ActionTypes, TaskResult
from llama_index.core.llms import ChatMessage, ChatResponseGen

from snowflake_cybersyn_demo.additional_services.human_in_the_loop import (
    human_service_factory,
)

logger = logging.getLogger(__name__)


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


class Controller:
    def __init__(
        self,
        human_in_loop_queue: asyncio.Queue,
        human_in_loop_result_queue: asyncio.Queue,
        submitted_tasks_queue: queue.Queue,
        control_plane_host: str = "127.0.0.1",
        control_plane_port: Optional[int] = 8000,
    ):
        self.human_in_loop_queue = human_in_loop_queue
        self.human_in_loop_result_queue = human_in_loop_result_queue
        self.submitted_tasks_queue: queue.Queue[TaskModel] = queue.Queue()
        self._human_service = human_service_factory(
            human_in_loop_queue, human_in_loop_result_queue
        )
        self._client = LlamaAgentsClient(
            control_plane_url=(
                f"http://{control_plane_host}:{control_plane_port}"
                if control_plane_port
                else f"http://{control_plane_host}"
            )
        )
        self._step_interval = 0.5
        self._timeout = 60
        self._raise_timeout = False
        self._human_in_the_loop_task: Optional[str] = None
        self._human_input: Optional[str] = None
        self._final_task_consumer = CallableMessageConsumer(
            message_type="human", handler=self._process_completed_task_messages
        )
        self._completed_tasks_queue: asyncio.Queue[TaskResult] = asyncio.Queue()

    async def _process_completed_task_messages(
        self, message: QueueMessage, **kwargs: Any
    ) -> None:
        """Consumer of completed tasks.

        By default control plane sends to message consumer of type "human".
        The process message logic contained here simply puts the TaskResult into
        a queue that is continuosly via a gr.Timer().
        """
        if message.action == ActionTypes.COMPLETED_TASK:
            task_res = TaskResult(**message.data)
            await self._completed_tasks_queue.put(task_res)
            logger.info("Added task result to queue")

    def _llama_index_stream_wrapper(
        self,
        llama_index_stream: ChatResponseGen,
    ) -> Generator[str, Any, Any]:
        for chunk in llama_index_stream:
            yield chunk.delta

    def _handle_task_submission(self) -> None:
        """Handle the user submitted message. Clear task submission box, and
        add the new task to the submitted list.
        """

        # create new task and store in state
        task_input = st.session_state.task_input
        if task_input == "":
            return
        task_id = self._client.create_task(task_input)
        task = TaskModel(
            task_id=task_id,
            input=task_input,
            chat_history=[
                ChatMessage(role="user", content=task_input),
                ChatMessage(
                    role="assistant",
                    content=f"Successfully submitted task: {task_id}.",
                ),
            ],
            status=TaskStatus.SUBMITTED,
        )
        st.session_state.submitted_tasks.append(task)
        logger.info("Added task to submitted queue")
        st.session_state.task_input = ""
