import asyncio
import logging
from typing import Any, Dict

from llama_agents import HumanService, ServiceComponent
from llama_agents.message_queues.rabbitmq import RabbitMQMessageQueue

from snowflake_cybersyn_demo.utils import load_from_env

logger = logging.getLogger("snowflake_cybersyn_demo")
logging.basicConfig(level=logging.INFO)

message_queue_host = load_from_env("RABBITMQ_HOST")
message_queue_port = load_from_env("RABBITMQ_NODE_PORT")
message_queue_username = load_from_env("RABBITMQ_DEFAULT_USER")
message_queue_password = load_from_env("RABBITMQ_DEFAULT_PASS")
control_plane_host = load_from_env("CONTROL_PLANE_HOST")
control_plane_port = load_from_env("CONTROL_PLANE_PORT")
human_in_the_loop_host = load_from_env("HUMAN_IN_THE_LOOP_HOST")
human_in_the_loop_port = load_from_env("HUMAN_IN_THE_LOOP_PORT")
localhost = load_from_env("LOCALHOST")


# # human in the loop function
def human_service_factory(
    human_input_request_queue: asyncio.Queue[Dict[str, str]],
    human_input_result_queue: asyncio.Queue[str],
):
    async def human_input_fn(prompt: str, task_id: str, **kwargs: Any) -> str:
        logger.info("human input fn invoked.")
        await human_input_request_queue.put(
            {"prompt": prompt, "task_id": task_id}
        )
        logger.info("placed new prompt in queue.")

        # poll until human answer is stored
        async def _poll_for_human_input_result() -> str:
            return await human_input_result_queue.get()

        try:
            human_input = await asyncio.wait_for(
                _poll_for_human_input_result(),
                timeout=6000,
            )
            logger.info(f"Recieved human input: {human_input}")
        except (
            asyncio.exceptions.TimeoutError,
            asyncio.TimeoutError,
            TimeoutError,
        ):
            logger.info(f"Timeout reached for tool_call with prompt {prompt}")
            human_input = "Something went wrong."

        return human_input

    # create our multi-agent framework components
    message_queue = RabbitMQMessageQueue(
        url=f"amqp://{message_queue_username}:{message_queue_password}@{message_queue_host}:{message_queue_port}/"
    )
    human_service = HumanService(
        message_queue=message_queue,
        description="Answers queries about math.",
        fn_input=human_input_fn,
        human_input_prompt="{input_str}",
    )
    return human_service


# used by control plane
human_service = human_service_factory(asyncio.Queue(), asyncio.Queue())
human_component = ServiceComponent.from_service_definition(human_service)
