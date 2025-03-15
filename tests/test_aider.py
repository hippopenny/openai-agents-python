import unittest
from unittest.mock import AsyncMock, patch
from examples.coder.aider import AiderConfig, AiderRunner, OutputProcessor, CoderAgent
from agents import Agent, MessageOutputItem

class TestAiderConfig(unittest.TestCase):
    def test_default_values(self):
        config = AiderConfig()
        self.assertEqual(config.repo_path, ".")
        self.assertEqual(config.model, "gpt-4")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.allow_dirty, True)
        self.assertEqual(config.auto_commit, True)

    def test_custom_values(self):
        config = AiderConfig(repo_path="/tmp/repo", model="gpt-3.5-turbo", temperature=0.5, allow_dirty=False, auto_commit=False)
        self.assertEqual(config.repo_path, "/tmp/repo")
        self.assertEqual(config.model, "gpt-3.5-turbo")
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
            stdout, stderr = await runner.execute(["aider", "--input", "test"])

            mock_exec.assert_called_once_with("aider", "--input", "test", stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(stdout, b"stdout_output")
            self.assertEqual(stderr, b"stderr_output")

class TestOutputProcessor(unittest.TestCase):
    def test_process_output_with_errors(self):
        result = OutputProcessor.process_output(b"some output", b"some error")
        self.assertEqual(result, {"output": "some output", "errors": "some error"})

    def test_process_output_without_errors(self):
        result = OutputProcessor.process_output(b"some output", b"")
        self.assertEqual(result, {"output": "some output", "errors": ""})

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

        expected_output = [MessageOutputItem.from_text("success output", role="assistant")]
        actual_output = await agent.run("test input")
        self.assertEqual(actual_output, expected_output)


    async def test_run_error(self):
        config = AiderConfig()
        agent = CoderAgent(name="test_agent", instructions="test instructions", handoff_description="test handoff", config=config)

        mock_runner = AsyncMock()
        mock_runner.execute.return_value = (b"", b"error output")  # Simulate error
        agent.aider_runner = mock_runner

        expected_output = [MessageOutputItem.from_text("Aider Error: error output", role="assistant")]
        actual_output = await agent.run("test input")
        self.assertEqual(actual_output, expected_output)
