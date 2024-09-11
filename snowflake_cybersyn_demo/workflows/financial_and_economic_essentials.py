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
        candidates = db.get_list_of_candidate_goods(good=good)
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
        timeseries_data_str = db.get_time_series_of_good(good=ev.selected_good)
        # aggregation
        aggregated_timeseries_data = db.perform_price_aggregation(
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
