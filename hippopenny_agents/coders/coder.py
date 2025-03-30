import os
from agents import Agent
from agents.models.hippopenny_aider_provider import HippoPennyAiderModelProvider
from .context import CoderContext
from .prompts import SystemPrompt


def create_coder_agent() -> Agent[CoderContext]:
    """
    Creates the Coder agent, configured to use the Aider model via the proxy.
    """

    coder_model_provider = HippoPennyAiderModelProvider()
    coder_model = coder_model_provider.get_model("aider") # aider proxy hardcodes model for now
    prompt = SystemPrompt()

    coder_agent = Agent[CoderContext](
        name="CoderAgent",
        model=coder_model,
        instructions=prompt.get_system_message(),
    )
    return coder_agent

# Example usage (optional, for testing)
if __name__ == "__main__":
    async def test_coder():
        from agents import Runner
        coder = create_coder_agent()
        # CoderContext isn't strictly needed by the coder itself if it only gets the task string,
        # but the Runner expects a context type.
        context = CoderContext(initial_request="Test")
        task_description = "Create a python function that prints 'hello world'"
        print(f"Running Coder Agent for task: {task_description}")
        # In real usage, the Planner's tool would call this.
        result = await Runner.run(coder, task_description, context=context)
        print("\nCoder Agent Result:")
        print(result.final_output)

    import asyncio
    asyncio.run(test_coder())
