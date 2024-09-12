from typing import Callable

from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step


class HumanInputWorkflow(Workflow):
    input: Callable = input

    @step
    async def human_input(self, ev: StartEvent) -> StopEvent:
        prompt = str(ev.get("prompt", ""))
        human_input = self.input(prompt)
        return StopEvent(result=human_input)


# Local Testing
async def _test_workflow() -> None:
    w = HumanInputWorkflow(timeout=None, verbose=False)
    result = await w.run(prompt="How old are you?\n\n")
    print(str(result))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_workflow())
