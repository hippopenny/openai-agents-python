from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from pydantic import ValidationError

# Re-introduce agents SDK dependency if Planner uses an Agent internally,
# or langchain if it calls an LLM directly. For now, it's a placeholder.
# from langchain_core.language_models.chat_models import BaseChatModel
# from langchain_core.messages import SystemMessage, HumanMessage

from .prompts import PlannerPromptBuilder
from .views import PlannerOutput

logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 3. Planner: Uses a planning prompt to generate a plan.
# (Based on high-level example, using PlannerPromptBuilder)
# ----------------------------------------------------------

class Planner:
    def __init__(self, llm: Optional[Any] = None, prompt: Optional[str] = None) -> None:
        """
        Initializes the Planner.

        Args:
            llm: The language model instance to use for generating plans.
                 If None, generate_plan will return a dummy plan.
            prompt: Custom system prompt string. If None, uses default from PlannerPromptBuilder.
        """
        self.llm = llm # Placeholder for the actual LLM client
        self.prompt_builder = PlannerPromptBuilder()
        self.system_prompt = prompt or self.prompt_builder.get_system_message_content()
        logger.info(f"Planner initialized. LLM provided: {self.llm is not None}")

    async def generate_plan(self, messages: List[str]) -> Dict[str, Any] | str:
        """
        Generates a plan based on the message history.

        Args:
            messages: List of string messages representing the history.

        Returns:
            A dictionary representing the parsed PlannerOutput JSON,
            or a string containing the raw plan or an error message.
        """
        if not self.llm:
            logger.warning("Planner LLM not provided. Returning dummy plan.")
            # Placeholder dummy plan if no LLM is available
            dummy_plan = {
                "state_analysis": "Initial state observed (dummy plan).",
                "progress_evaluation": "0% complete (dummy plan).",
                "challenges": ["LLM not configured for planner."],
                "next_steps": ["Configure planner LLM", "Provide real context"],
                "reasoning": "This is a placeholder plan generated because no LLM was provided to the Planner.",
            }
            return dummy_plan

        logger.info("Generating plan using Planner LLM...")
        # In a full implementation, combine self.system_prompt with the messages
        # and call the LLM. This requires adapting string messages to the LLM's expected format.
        # Example using langchain_core style (requires llm to be BaseChatModel):
        # planner_messages = [SystemMessage(content=self.system_prompt)]
        # planner_messages.extend([HumanMessage(content=msg) for msg in messages]) # Simplistic conversion
        # try:
        #     response = await self.llm.ainvoke(planner_messages)
        #     raw_plan_content = response.content
        # except Exception as e:
        #     logger.error(f"Planner LLM invocation failed: {e}", exc_info=True)
        #     return f"Error during planning LLM call: {e}"

        # Placeholder LLM call simulation:
        await asyncio.sleep(0.2) # Simulate async work
        # Simulate receiving a JSON string (potentially malformed)
        raw_plan_content = json.dumps({
                "state_analysis": "Analyzed history (simulated LLM response).",
                "progress_evaluation": "Simulated 25% progress.",
                "challenges": ["Potential need for scrolling."],
                "next_steps": ["Simulated: Click button 'Login'", "Simulated: Input username"],
                "reasoning": "Simulated reasoning based on history.",
        })
        logger.debug(f"Planner raw output (simulated): {raw_plan_content}")


        # Attempt to parse the JSON output
        try:
            # Basic JSON parsing first
            parsed_json = json.loads(raw_plan_content)
            # Validate against the PlannerOutput model
            plan_output = PlannerOutput.model_validate(parsed_json)
            logger.info("Planner output parsed and validated successfully.")
            return plan_output.model_dump() # Return as dict
        except json.JSONDecodeError:
            logger.warning(f"Planner output was not valid JSON: {raw_plan_content[:200]}...")
            return raw_plan_content # Return raw string
        except ValidationError as e:
            logger.warning(f"Planner output failed Pydantic validation: {e}. Returning raw content.")
            return raw_plan_content # Return raw string
        except Exception as e:
            logger.error(f"Unexpected error parsing planner output: {e}", exc_info=True)
            return f"Error parsing planner output: {e}"

