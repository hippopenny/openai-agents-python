from typing import Literal
from pydantic import BaseModel, Field

from agents import Agent

# Define CoderOutput here as well for clarity, though it's also in models.py
class CoderOutput(BaseModel):
    status: Literal["completed", "failed", "needs_clarification"]
    summary: str = Field(description="Summary of the work done or explanation of failure/clarification needed.")
    code_changes: str | None = Field(default=None, description="Actual code changes or implementation details.")


# Note: The actual implementation logic (code generation) is handled by the LLM based on the prompt.
# This agent definition primarily sets up the instructions and expected output format.
CoderAgent = Agent(
    name="CoderAgent",
    instructions=(
        "You are a skilled software engineer. You will be given a single, specific task description. "
        "Your goal is to implement this task. "
        "Output your results using the CoderOutput format. "
        "If you successfully implement the task, set status to 'completed' and include the code or a description of changes in 'code_changes'. "
        "If you encounter an error you cannot resolve, set status to 'failed' and explain the error in 'summary'. "
        "If the task description is unclear or requires more information, set status to 'needs_clarification' and ask clarifying questions in 'summary'."
    ),
    output_type=CoderOutput,
    # Consider using a more capable model for coding tasks
    # model="gpt-4-turbo",
)

