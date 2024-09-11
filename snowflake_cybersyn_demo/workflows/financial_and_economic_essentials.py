from typing import List

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI
from snowflake.sqlalchemy import URL
from sqlalchemy import create_engine, text

from snowflake_cybersyn_demo.utils import load_from_env
from snowflake_cybersyn_demo.workflows.sql_queries import (
    get_time_series_of_good,
    perform_price_aggregation,
)

snowflake_user = load_from_env("SNOWFLAKE_USERNAME")
snowflake_password = load_from_env("SNOWFLAKE_PASSWORD")
snowflake_account = load_from_env("SNOWFLAKE_ACCOUNT")
snowflake_role = load_from_env("SNOWFLAKE_ROLE")

SQL_QUERY_TEMPLATE = """
SELECT DISTINCT att.product,
FROM cybersyn.bureau_of_labor_statistics_price_timeseries AS ts
JOIN cybersyn.bureau_of_labor_statistics_price_attributes AS att
    ON (ts.variable = att.variable)
WHERE ts.date >= '2021-01-01'
  AND att.report = 'Average Price'
  AND att.product ILIKE '{good}%';
"""


def get_list_of_candidate_goods(good: str) -> List[str]:
    """Returns a list of goods that exist in the database.

    The list of goods is represented as a string separated by '\n'."""
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

    return [f"{ix+1}. {str(el[0])}" for ix, el in enumerate(results)]


class CandidateLookupEvent(Event):
    candidates: List[str]


class HumanInputEvent(Event):
    input: str
    selected_good: str


class GoodsTimeSeriesWorkflow(Workflow):
    @step
    async def retrieve_candidates_from_db(
        self, ev: StartEvent
    ) -> CandidateLookupEvent:
        # Your workflow logic here
        good = str(ev.get("good", ""))
        candidates = get_list_of_candidate_goods(good=good)
        return CandidateLookupEvent(candidates=candidates)

    @step
    async def human_input(self, ev: CandidateLookupEvent) -> HumanInputEvent:
        candidate_list = "\n".join(ev.candidates)
        human_prompt = (
            "List of goods that exist in the database are provided below."
            f"{candidate_list}"
            "\n\nPlease select one.:\n\n"
        )
        human_input = input(human_prompt)

        # use llm to clean up selection
        llm = OpenAI("gpt-4o")
        llm_prompt = (
            "Below we provide a list of goods as well as a human's selection from this list."
            "LIST OF GOODS:\n\n"
            f"{candidate_list}"
            "\n\n"
            "HUMAN SELECTION:\n\n"
            f"{human_input}"
            "\n\n"
            "Return the good that the human selected without its item number. An example is provided below:"
            "\n\n"
            "LIST OF GOODS:\n\n1. ABC\n2. DEF\n\nHUMAN SELECTION: 2\n\nDEF"
        )
        llm_response = await llm.acomplete(prompt=llm_prompt)
        return HumanInputEvent(
            input=human_input, selected_good=llm_response.text
        )

    @step
    async def get_time_series_data(self, ev: HumanInputEvent) -> StopEvent:
        timeseries_data_str = get_time_series_of_good(good=ev.selected_good)
        # aggregation
        aggregated_timeseries_data = perform_price_aggregation(
            timeseries_data_str
        )
        return StopEvent(result=aggregated_timeseries_data)


# Local Testing
async def _test_workflow() -> None:
    w = GoodsTimeSeriesWorkflow(timeout=None, verbose=False)
    result = await w.run(input="gasoline")
    print(str(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
