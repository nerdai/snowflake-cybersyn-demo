from typing import List

from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    step,
)
from llama_index.llms.openai import OpenAI

import snowflake_cybersyn_demo.workflows._db as db


class StatisticsLookupEvent(Event):
    statistic_variables: List[str]
    city: str


class HumanInputEvent(Event):
    input: str
    selected_stat: str
    city: str


class GovtEssentialsStatisticsWorkflow(Workflow):
    @step
    async def retrieve_candidates_from_db(
        self, ev: StartEvent
    ) -> StatisticsLookupEvent:
        # Your workflow logic here
        city = str(ev.get("city", ""))
        stats_vars = db.get_list_of_statistical_variables(city=city)
        return StatisticsLookupEvent(statistic_variables=stats_vars, city=city)

    @step
    async def human_input(self, ev: StatisticsLookupEvent) -> HumanInputEvent:
        stats_vars = "\n".join(ev.statistic_variables)
        human_prompt = (
            "List of statistic variables that exist in the database are provided below."
            f"{stats_vars}"
            "\n\nPlease select one.:\n\n"
        )
        human_input = input(human_prompt)

        # use llm to clean up selection
        llm = OpenAI("gpt-4o")
        llm_prompt = (
            "Below we provide a list of statistics as well as a human's selection from this list."
            "LIST OF STATISTICS:\n\n"
            f"{stats_vars}"
            "\n\n"
            "HUMAN SELECTION:\n\n"
            f"{human_input}"
            "\n\n"
            "Return the statistic that the human selected without its item number. An example is provided below:"
            "\n\n"
            "LIST OF STATISTICS:\n\n1. ABC\n2. DEF\n\nHUMAN SELECTION: 2\n\nDEF"
        )
        llm_response = await llm.acomplete(prompt=llm_prompt)
        return HumanInputEvent(
            input=human_input, selected_stat=llm_response.text, city=ev.city
        )

    @step
    async def get_time_series_data(self, ev: HumanInputEvent) -> StopEvent:
        timeseries_data_str = db.get_time_series_of_statistic_variable(
            city=ev.city, stats_variable=ev.selected_stat
        )
        # aggregation
        aggregated_timeseries_data = db.perform_date_value_aggregation(
            timeseries_data_str
        )
        return StopEvent(result=aggregated_timeseries_data)


# Local Testing
async def _test_workflow() -> None:
    w = GovtEssentialsStatisticsWorkflow(timeout=None, verbose=False)
    result = await w.run(city="New York")
    print(str(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
