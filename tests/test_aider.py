import unittest
from unittest.mock import AsyncMock, patch
import subprocess
import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from examples.coder.aider import AiderConfig, AiderRunner, OutputProcessor, CoderAgent
from agents import Agent, MessageOutputItem, ItemHelpers


class TestAiderConfig(unittest.TestCase):
    def test_default_values(self):
        config = AiderConfig()
        self.assertEqual(config.repo_path, ".")
        self.assertEqual(config.model, "openrouter/google/gemini-2.0-pro-exp-02-05:free")
        self.assertEqual(config.editor_model, "gemini/gemini-2.0-flash-exp")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.allow_dirty, True)
        self.assertEqual(config.auto_commit, True)

    def test_custom_values(self):
        config = AiderConfig(repo_path="/tmp/repo", model="gpt-3.5-turbo", editor_model="another-model", temperature=0.5, allow_dirty=False, auto_commit=False)
        self.assertEqual(config.repo_path, "/tmp/repo")
        self.assertEqual(config.model, "gpt-3.5-turbo")
        self.assertEqual(config.editor_model, "o1-mini")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.allow_dirty, False)
        self.assertEqual(config.auto_commit, False)

class TestAiderRunner(unittest.IsolatedAsyncioTestCase):
    async def test_execute(self):
        config = AiderConfig()
        runner = AiderRunner(config)

        mock_process = AsyncMock()
        mock_process.communicate.return_value = (b"stdout_output", b"stderr_output")

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            stdout, stderr = await runner.execute(["aiderhp", "--input", "test", "--model", config.model, "--editor-model", config.editor_model, "--repo", config.repo_path])

            mock_exec.assert_called_once_with("aiderhp", "--input", "test", "--model", config.model, "--editor-model", config.editor_model, "--repo", config.repo_path, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(stdout, b"stdout_output")
            self.assertEqual(stderr, b"stderr_output")

class TestOutputProcessor(unittest.TestCase):
    def test_process_output_with_errors(self):
        result = OutputProcessor.process_output(b"some output", b"some error")
        self.assertEqual(result, {"output": "some output", "errors": "some error"})

    def test_process_output_without_errors(self):
        result = OutputProcessor.process_output(b"some output", b"")
        self.assertEqual(result, {"output": "some output", "errors": None})

    def test_process_output_with_no_errors(self):
        result = OutputProcessor.process_output(b"some output", None)
        self.assertEqual(result, {"output": "some output", "errors": None})

class TestCoderAgent(unittest.IsolatedAsyncioTestCase):
    async def test_run_success(self):
        config = AiderConfig()
        agent = CoderAgent(name="test_agent", instructions="test instructions", handoff_description="test handoff", config=config)

        mock_runner = AsyncMock()
        mock_runner.execute.return_value = (b"success output", b"")  # Simulate successful execution
        agent.aider_runner = mock_runner

        # Call modify_code directly
        actual_output = await agent.modify_code("test input")
        self.assertEqual(actual_output, "success output")


    async def test_run_error(self):
        config = AiderConfig()
        agent = CoderAgent(name="test_agent", instructions="test instructions", handoff_description="test handoff", config=config)

        mock_runner = AsyncMock()
        mock_runner.execute.return_value = (b"", b"error output")  # Simulate error
        agent.aider_runner = mock_runner

        # Call modify_code directly
        actual_output = await agent.modify_code("test input")
        self.assertEqual(actual_output, "Aider Error: error output")

    async def test_hello_world(self):
        config = AiderConfig()
        agent = CoderAgent(name="test_agent", instructions="test instructions", handoff_description="test handoff", config=config)

        mock_runner = AsyncMock()
        mock_runner.execute.return_value = (b"ADDED: hello.py\n", b"")  # Simulate aider output
        agent.aider_runner = mock_runner

        actual_output = await agent.modify_code("Create a new file hello.py and write 'hello world' to it.")
        self.assertEqual(actual_output, "ADDED: hello.py\n")
