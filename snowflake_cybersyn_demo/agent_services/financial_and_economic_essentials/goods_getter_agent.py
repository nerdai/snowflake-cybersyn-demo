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
localhost = load_from_env("LOCALHOST")


# create agent server
message_queue = RabbitMQMessageQueue(
    url=f"amqp://{message_queue_username}:{message_queue_password}@{message_queue_host}:{message_queue_port}/"
)

SQL_QUERY_TEMPLATE = """
SELECT DISTINCT att.product,
FROM cybersyn.bureau_of_labor_statistics_price_timeseries AS ts
JOIN cybersyn.bureau_of_labor_statistics_price_attributes AS att
    ON (ts.variable = att.variable)
WHERE ts.date >= '2021-01-01'
  AND att.report = 'Average Price'
  AND att.product ILIKE '{good}%';
"""

AGENT_SYSTEM_PROMPT = """
For a given query about a good in the database, your job is to first find
if the good exists in the database. Return the list of goods in the database
that potentially match the object of the users query.
"""


def get_list_of_candidate_goods(good: str) -> str:
    """Returns a list of goods that exist in the database.

    The list of goods is represented as a string separated by '\n'."""
    query = SQL_QUERY_TEMPLATE.format(good=good)
    url = URL(
        account="AZXOMEC-NZB11223",
        user="NERDAILLAMAINDEX",
        password="b307gJ5YzR8k",
        database="FINANCIAL__ECONOMIC_ESSENTIALS",
        schema="CYBERSYN",
        warehouse="COMPUTE_WH",
        role="ACCOUNTADMIN",
    )

    engine = create_engine(url)
    try:
        connection = engine.connect()
        results = connection.execute(text(query))
    finally:
        connection.close()

    # process
    results = [str(el[0]) for el in results]
    results_str = "\n".join(results)

    return results_str


goods_getter_tool = FunctionTool.from_defaults(fn=get_list_of_candidate_goods)
agent = OpenAIAgent.from_tools(
    [goods_getter_tool],
    system_prompt=AGENT_SYSTEM_PROMPT,
    llm=OpenAI(model="gpt-4o-mini"),
    verbose=True,
)
