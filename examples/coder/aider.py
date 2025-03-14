import asyncio
import subprocess

from agents import Agent, ItemHelpers, MessageOutputItem, Runner, trace

"""
This workflow shows the agents-as-tools pattern. The frontline agent receives a user message and
then picks which agents to call, as tools.
"""

class CoderAgent(Agent):
    def __init__(self, name: str, instructions: str, handoff_description: str, repo_path: str = "."):
        super().__init__(name=name, instructions=instructions)
        self.handoff_description = handoff_description
        self.repo_path = repo_path # Path to the git repository

    async def run(self, input_message):
        # 1. Prepare the command to run aider.  This is a placeholder
        #    and will need to be adapted based on the actual aider CLI.
        command = [
            "aider",
            "--input", input_message,
            "--repo", self.repo_path
            # ... other aider options ...
        ]

        # 2. Run aider as a subprocess.
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # 3. Get the output and error streams.
        stdout, stderr = await process.communicate()

        # 4. Check for errors.
        if process.returncode != 0:
            error_message = stderr.decode()
            # Handle the error (e.g., return an error message, raise an exception)
            return [MessageOutputItem.from_text(f"Aider Error: {error_message}", role="assistant")]

        # 5. Process the output.
        output_text = stdout.decode()
        return [MessageOutputItem.from_text(output_text, role="assistant")]

    def as_tool(self, tool_name: str, tool_description: str):
        return {
            "name": tool_name,
            "description": tool_description,
            "agent": self,
        }

    def __repr__(self):
        return f"CoderAgent(name={self.name}, instructions={self.instructions}, handoff_description={self.handoff_description})"

# --- Example Usage (within the larger example) ---

coder_agent = CoderAgent(
    name="coder_agent",
    instructions="You are a coding assistant. You use aider to modify code in the repository.",
    handoff_description="A coding assistant that uses aider.",
    repo_path="."  # Assuming the current directory is the repo
)

orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions=(
        "You are a coding assistant orchestrator. You use the tools given to you, including the coder_agent, to help the user."
    ),
    tools=[
        coder_agent.as_tool(
            tool_name="modify_code",
            tool_description="Use aider to modify code in the repository based on the user's instructions.",
        ),
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
