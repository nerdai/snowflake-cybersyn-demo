import asyncio

import uvicorn
from llama_agents import ControlPlaneServer, PipelineOrchestrator
from llama_agents.message_queues.apache_kafka import KafkaMessageQueue
from llama_index.core.query_pipeline import QueryPipeline, RouterComponent
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.additional_services.human_in_the_loop import (
    human_component,
    human_service,
)
from snowflake_cybersyn_demo.agent_services.funny_agent import (
    agent_component,
    agent_server,
)
from snowflake_cybersyn_demo.utils import load_from_env

message_queue_host = load_from_env("KAFKA_HOST")
message_queue_port = load_from_env("KAFKA_PORT")
control_plane_host = load_from_env("CONTROL_PLANE_HOST")
control_plane_port = load_from_env("CONTROL_PLANE_PORT")
localhost = load_from_env("LOCALHOST")


# setup message queue
message_queue = KafkaMessageQueue.from_url_params(
    host=message_queue_host,
    port=int(message_queue_port) if message_queue_port else None,
)

pipeline = QueryPipeline(
    chain=[
        RouterComponent(
            selector=PydanticSingleSelector.from_defaults(llm=OpenAI()),
            choices=[agent_server.description, human_service.description],
            components=[agent_component, human_component],
        )
    ]
)
pipeline_orchestrator = PipelineOrchestrator(pipeline)

# setup control plane
control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=pipeline_orchestrator,
    host=control_plane_host,
    port=int(control_plane_port) if control_plane_port else None,
)


app = control_plane.app


# launch
async def launch() -> None:
    # register to message queue and start consuming
    start_consuming_callable = await control_plane.register_to_message_queue()
    _ = asyncio.create_task(start_consuming_callable())

    cfg = uvicorn.Config(
        control_plane.app,
        host=localhost,
        port=control_plane.port,
    )
    server = uvicorn.Server(cfg)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(launch())
