import asyncio
import subprocess
from dataclasses import dataclass
from typing import Optional, List

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace, function_tool

"""
This workflow shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools.
"""

@dataclass
class AiderConfig:
    repo_path: str = "."
    model: str = "openrouter/google/gemini-2.0-pro-exp-02-05:free"
    editor_model: str = "gemini/gemini-2.0-flash-exp"
    temperature: float = 0.7 # Placeholder
    allow_dirty: bool = True # Placeholder
    auto_commit: bool = True # Placeholder
    api_key: str | None = None
    api_base: str | None = None


class AiderRunner:
    def __init__(self, config: AiderConfig):
        self.config = config

    async def execute(self, command: List[str]) -> tuple[bytes, bytes]:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return await process.communicate()

class OutputProcessor:
    @staticmethod
    def process_output(stdout: bytes, stderr: bytes) -> dict:
        return {
            "output": stdout.decode(),
            "errors": stderr.decode() if stderr else None
        }

class CoderAgent(Agent):
    def __init__(self, name: str, instructions: str, handoff_description: str, config: AiderConfig):
        super().__init__(name=name, instructions=instructions)
        self.handoff_description = handoff_description
        self.aider_runner = AiderRunner(config)
        self.config = config

    @function_tool(name_override="modify_code", description_override="Use aider to modify code in the repository based on the user's instructions.")
    async def modify_code(self, instructions: str) -> str:
        """
        This method is the tool that will be called by other agents.
        It takes the user's instructions and runs aider.
        """
        # 1. Prepare the command to run aider.
        command = [
            "aiderhp",
            "--input", instructions,
            "--model", self.config.model,
            "--editor-model", self.config.editor_model,
            "--repo", self.config.repo_path,
            "--yes" # Add --yes for non-interactive use
        ]

        # 2. Run aider using AiderRunner.
        stdout, stderr = await self.aider_runner.execute(command)

        # 3. Process the output.
        processed_output = OutputProcessor.process_output(stdout, stderr)

        # 4. Check for errors.
        if processed_output["errors"]:
            # Handle the error
            return f"Aider Error: {processed_output['errors']}"

        # 5. Return the successful output.
        return processed_output["output"]

    def __repr__(self):
        return f"CoderAgent(name={self.name}, instructions={self.instructions}, handoff_description={self.handoff_description})"

# --- Example Usage ---

coder_config = AiderConfig(repo_path=".")  # Use default config

coder_agent = CoderAgent(
    name="coder_agent",
    instructions="You are a coding assistant. You use aider to modify code in the repository.",
    handoff_description="A coding assistant that uses aider.",
    config=coder_config,
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a coding assistant orchestrator. You use the tools given to you, including the coder_agent, to help the user."
    ),
    tools=[
        coder_agent.modify_code,
    ],
)

async def main():
    msg = input("Hi! What would you like me to do with the code? ")

    with trace("Orchestrator evaluator"):
        orchestrator_result = await Runner.run(orchestrator_agent, msg)

        for item in orchestrator_result.new_items:
            if isinstance(item, MessageOutputItem):
                text = ItemHelpers.text_message_output(item)
                if text:
                    print(f"  - Step: {text}")

    print(f"\n\nFinal response:\n{orchestrator_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
