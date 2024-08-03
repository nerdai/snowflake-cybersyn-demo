import asyncio

import uvicorn
from llama_agents import ControlPlaneServer, PipelineOrchestrator
from llama_agents.orchestrators.router import RouterOrchestrator
from llama_agents.message_queues.rabbitmq import RabbitMQMessageQueue
from llama_index.core.query_pipeline import QueryPipeline, RouterComponent
from llama_index.core.selectors import PydanticSingleSelector
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.additional_services.human_in_the_loop import (
    human_component,
)
from snowflake_cybersyn_demo.agent_services import (
    funny_agent_component,
    funny_agent_server,
    goods_getter_agent_component,
    time_series_getter_agent_component,
)
from snowflake_cybersyn_demo.utils import load_from_env

message_queue_host = load_from_env("RABBITMQ_HOST")
message_queue_port = load_from_env("RABBITMQ_NODE_PORT")
message_queue_username = load_from_env("RABBITMQ_DEFAULT_USER")
message_queue_password = load_from_env("RABBITMQ_DEFAULT_PASS")
control_plane_host = load_from_env("CONTROL_PLANE_HOST")
control_plane_port = load_from_env("CONTROL_PLANE_PORT")
localhost = load_from_env("LOCALHOST")


# setup message queue
message_queue = RabbitMQMessageQueue(
    url=f"amqp://{message_queue_username}:{message_queue_password}@{message_queue_host}:{message_queue_port}/"
)


timeseries_task_pipeline = QueryPipeline(
    chain=[
        goods_getter_agent_component,
        human_component,
        time_series_getter_agent_component,
    ],
)
timeseries_pipeline_orchestrator = PipelineOrchestrator(timeseries_task_pipeline)
timeseries_task_pipeline_desc = (
    "Only used for getting timeseries data from the database."
)

general_pipeline = QueryPipeline(chain=[funny_agent_component])
general_pipeline_orchestrator = PipelineOrchestrator(general_pipeline)

pipeline_orchestrator = RouterOrchestrator(
    selector=PydanticSingleSelector.from_defaults(llm=OpenAI()),
    orchestrators=[timeseries_pipeline_orchestrator, general_pipeline_orchestrator],
    choices=[timeseries_task_pipeline_desc, funny_agent_server.description],
)

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
