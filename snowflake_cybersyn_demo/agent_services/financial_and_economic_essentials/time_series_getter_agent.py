import json
from typing import Dict, List

from llama_agents.message_queues.rabbitmq import RabbitMQMessageQueue
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, text

from snowflake_cybersyn_demo.utils import load_from_env

message_queue_host = load_from_env("RABBITMQ_HOST")
message_queue_port = load_from_env("RABBITMQ_NODE_PORT")
message_queue_username = load_from_env("RABBITMQ_DEFAULT_USER")
message_queue_password = load_from_env("RABBITMQ_DEFAULT_PASS")
control_plane_host = load_from_env("CONTROL_PLANE_HOST")
control_plane_port = load_from_env("CONTROL_PLANE_PORT")
funny_agent_host = load_from_env("FUNNY_AGENT_HOST")
funny_agent_port = load_from_env("FUNNY_AGENT_PORT")
snowflake_user = load_from_env("SNOWFLAKE_USERNAME")
snowflake_password = load_from_env("SNOWFLAKE_PASSWORD")
snowflake_account = load_from_env("SNOWFLAKE_ACCOUNT")
snowflake_role = load_from_env("SNOWFLAKE_ROLE")
localhost = load_from_env("LOCALHOST")


# create agent server
message_queue = RabbitMQMessageQueue(
    url=f"amqp://{message_queue_username}:{message_queue_password}@{message_queue_host}:{message_queue_port}/"
)

AGENT_SYSTEM_PROMPT = """
Query the database to return timeseries data of user-specified good.

Use the tool to return the time series data as a JSON with the folowing format:

{{
    [
        {{
            "good": ...,
            "date": ...,
            "price": ...
        }},
        {{
            "good": ...,
            "date": ...,
            "price": ...
        }},
        ...
    ]
}}

Don't return the output as markdown code. Don't modify the tool output. Return
strictly the tool ouput.
"""

SQL_QUERY_TEMPLATE = """
SELECT ts.date,
       att.variable_name,
       ts.value
FROM cybersyn.bureau_of_labor_statistics_price_timeseries AS ts
JOIN cybersyn.bureau_of_labor_statistics_price_attributes AS att
    ON (ts.variable = att.variable)
WHERE ts.date >= '2021-01-01'
  AND att.report = 'Average Price'
  AND att.product ILIKE '{good}%'
ORDER BY date;
"""


def get_time_series_of_good(good: str) -> str:
    """Create a time series of the average price paid for a good nationwide starting in 2021."""
    query = SQL_QUERY_TEMPLATE.format(good=good)
    url = URL(
        account=snowflake_account,
        user=snowflake_user,
        password=snowflake_password,
        database="FINANCIAL__ECONOMIC_ESSENTIALS",
        schema="CYBERSYN",
        warehouse="COMPUTE_WH",
        role=snowflake_role,
    )

    engine = create_engine(url)
    try:
        connection = engine.connect()
        results = connection.execute(text(query))
    finally:
        connection.close()

    # process
    results = [
        {"good": str(el[1]), "date": str(el[0]), "price": str(el[2])}
        for el in results
    ]
    results_str = json.dumps(results, indent=4)

    return results_str


def perform_price_aggregation(json_str: str) -> str:
    """Perform price aggregation on the time series data."""
    timeseries_data = json.loads(json_str)
    good = timeseries_data[0]["good"]

    new_time_series_data: Dict[str, List[float]] = {}
    for el in timeseries_data:
        date = el["date"]
        price = el["price"]
        if date in new_time_series_data:
            new_time_series_data[date].append(float(price))
        else:
            new_time_series_data[date] = [float(price)]

    reduced_time_series_data = [
        {"good": good, "date": date, "price": sum(prices) / len(prices)}
        for date, prices in new_time_series_data.items()
    ]

    return json.dumps(reduced_time_series_data, indent=4)


goods_getter_tool = FunctionTool.from_defaults(
    fn=get_time_series_of_good, return_direct=True
)
agent = OpenAIAgent.from_tools(
    [goods_getter_tool],
    system_prompt=AGENT_SYSTEM_PROMPT,
    llm=OpenAI(model="gpt-3.5-turbo"),
    verbose=True,
)