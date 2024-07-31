import json
from typing import Dict, List

from llama_agents.message_queues.rabbitmq import RabbitMQMessageQueue
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

from snowflake_cybersyn_demo.utils import load_from_env

message_queue_host = load_from_env("RABBITMQ_HOST")
message_queue_port = load_from_env("RABBITMQ_NODE_PORT")
message_queue_username = load_from_env("RABBITMQ_DEFAULT_USER")
message_queue_password = load_from_env("RABBITMQ_DEFAULT_PASS")
control_plane_host = load_from_env("CONTROL_PLANE_HOST")
control_plane_port = load_from_env("CONTROL_PLANE_PORT")
funny_agent_host = load_from_env("FUNNY_AGENT_HOST")
funny_agent_port = load_from_env("FUNNY_AGENT_PORT")
localhost = load_from_env("LOCALHOST")


# create agent server
message_queue = RabbitMQMessageQueue(
    url=f"amqp://{message_queue_username}:{message_queue_password}@{message_queue_host}:{message_queue_port}/"
)

AGENT_SYSTEM_PROMPT = """
Perform price aggregation on the time series data to ensure that each date only
has one associated price.

Return the time series data as a JSON with the folowing format:

{{
    [
        {{
            "good": ...,
            "date": ...,
            "price": ...
        }}
    ]
}}

Don't return the output as markdown code.
"""


def perform_price_aggregation(json_str: str) -> str:
    """Perform price aggregation on the time series data."""
    timeseries_data = json.loads(json_str)
    good = timeseries_data[0]["good"]

    new_time_series_data: Dict[str, List[float]] = {}
    for el in timeseries_data:
        date = el["date"]
        price = el["price"]
        if date in new_time_series_data:
            new_time_series_data[date].append(price)
        else:
            new_time_series_data[date] = [price]

    reduced_time_series_data = [
        {"good": good, "date": date, "price": sum(prices) / len(prices)}
        for date, prices in new_time_series_data.items()
    ]

    return json.dumps(reduced_time_series_data, indent=4)


price_aggregation_tool = FunctionTool.from_defaults(
    fn=perform_price_aggregation, return_direct=True
)
agent = OpenAIAgent.from_tools(
    [price_aggregation_tool],
    system_prompt=AGENT_SYSTEM_PROMPT,
    llm=OpenAI(model="gpt-3.5-turbo"),
    verbose=True,
)
