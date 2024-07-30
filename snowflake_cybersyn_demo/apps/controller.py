import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generator, List, Optional

import streamlit as st
from llama_agents import LlamaAgentsClient
from llama_index.core.llms import ChatMessage, ChatResponseGen

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
        control_plane_host: str = "127.0.0.1",
        control_plane_port: Optional[int] = 8000,
    ):
        self._client = LlamaAgentsClient(
            control_plane_url=(
                f"http://{control_plane_host}:{control_plane_port}"
                if control_plane_port
                else f"http://{control_plane_host}"
            )
        )
        self._step_interval = 0.5
        self._timeout = 60

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
