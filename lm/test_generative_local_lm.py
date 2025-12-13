"""Comprehensive unit tests for GenerativeLocalVLLM class.

We include both unit tests with mocked dependencies and integration tests that require a GPU.

Response format for MockGenerativeLocalVLLM (strict):
    list[list[list[str]]] = layers[requests[completions]]

Expected usage:

```bash
pytest lm/test_generative_local_lm.py -vv
```
"""

# Standard library imports
import logging
import os
from collections.abc import Generator

# Third-party imports
import pytest
import torch
from vllm import SamplingParams

# Local imports
from constants import Verbosity
from lm.generative_local_lm import (
    ChatCompletionResponse,
    Choice,
    GenerativeLocalVLLM,
    Usage,
)
from lm.lm_constants import MessageKey, MessageRole
from utilities_for_tests import MockGenerativeLocalVLLM

logger = logging.getLogger(__name__)


# =============================================================================
# Unit Tests (Mocked - No GPU Required)
# =============================================================================


class TestMockGenerativeLocalVLLMBasics:
    """Test basic MockGenerativeLocalVLLM functionality."""

    def test_mock_initialization(self) -> None:
        """Test that MockGenerativeLocalVLLM initializes without errors."""
        # Format: layers[requests[completions]] - 1 layer, 1 request, 1 completion
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Hello, world!"]]])
        assert mock_lm.model_name == "mock-generative-model"
        assert mock_lm.model_path == "mock-generative-model"
        assert mock_lm.history == []

    def test_mock_with_string_response(self) -> None:
        """Test MockGenerativeLocalVLLM with a simple string response."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test response"]]])
        response = mock_lm.forward(prompt="Hello")
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Test response"

    def test_mock_with_list_response(self) -> None:
        """Test MockGenerativeLocalVLLM with multiple choices."""
        # Format: 1 layer, 1 request, 2 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Response 1", "Response 2"]]])
        response = mock_lm.forward(
            prompt="Hello",
            sampling_params=SamplingParams(n=2),
        )
        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 2
        assert response.choices[0].message.content == "Response 1"
        assert response.choices[1].message.content == "Response 2"


class TestGenerativeLocalVLLMForward:
    """Test forward method for single inputs using MockGenerativeLocalVLLM."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance.

        Returns:
            A MockGenerativeLocalVLLM instance with a simple response.
        """
        # Format: layers[requests[completions]]
        return MockGenerativeLocalVLLM(responses=[[["Generated response"]]])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompt",
            "messages",
            "expected_content",
        ],
        # Parameter values
        [
            pytest.param(
                "Hello",                            # prompt
                None,                               # messages
                "Generated response",               # expected_content
                id="forward_with_prompt",
            ),
            pytest.param(
                None,                               # prompt
                [                                   # messages
                    {MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "Hello"}
                ],
                "Generated response",               # expected_content
                id="forward_with_messages",
            ),
        ],
    )
    def test_forward_success(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        expected_content: str,
    ) -> None:
        """Test forward method returns ChatCompletionResponse.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompt: Prompt string to test.
            messages: Messages to test.
            expected_content: Expected content in the response.
        """
        response = mock_lm.forward(prompt=prompt, messages=messages)

        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1
        assert response.choices[0].message.content == expected_content

    def test_forward_with_sampling_params(self, mock_lm: MockGenerativeLocalVLLM) -> None:
        """Test forward method with custom sampling parameters.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=100)
        response = mock_lm.forward(prompt="Hello", sampling_params=sampling_params)

        assert isinstance(response, ChatCompletionResponse)
        assert len(response.choices) == 1


class TestGenerativeLocalVLLMBatch:
    """Test batch method for multiple inputs using MockGenerativeLocalVLLM."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance for batch testing.

        Returns:
            A MockGenerativeLocalVLLM instance with batch responses.
        """
        # Format: 1 layer, 2 requests, 1 completion each
        return MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompts",
            "messages",
            "expected_count",
        ],
        # Parameter values
        [
            pytest.param(
                ["Hello", "Hi"],                    # prompts
                None,                               # messages
                2,                                  # expected_count
                id="batch_with_prompts",
            ),
            pytest.param(
                None,                               # prompts
                [                                   # messages
                    [
                        {MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "Hello"}
                    ],
                    [
                        {MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "Hi"}
                    ],
                ],
                2,                                  # expected_count
                id="batch_with_messages",
            ),
        ],
    )
    def test_batch_success(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompts: list[str] | None,
        messages: list[list[dict[str, str]]] | None,
        expected_count: int,
    ) -> None:
        """Test batch method returns list of ChatCompletionResponse.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompts: Prompts to test.
            messages: Messages to test.
            expected_count: Expected number of responses.
        """
        responses = mock_lm.batch(prompts=prompts, messages=messages)

        assert isinstance(responses, list)
        assert len(responses) == expected_count
        for response in responses:
            assert isinstance(response, ChatCompletionResponse)

    def test_batch_with_sampling_params_list(self) -> None:
        """Test batch method with list of sampling parameters."""
        # Format: 1 layer, 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])
        sampling_params = [
            SamplingParams(temperature=0.5),
            SamplingParams(temperature=0.9),
        ]
        responses = mock_lm.batch(prompts=["Hello", "Hi"], sampling_params=sampling_params)

        assert len(responses) == 2


class TestGenerativeLocalVLLMCall:
    """Test __call__ method dispatching using MockGenerativeLocalVLLM."""

    def test_call_with_prompt(self) -> None:
        """Test __call__ with prompt dispatches to forward."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Hello response"]]])
        response = mock_lm(prompt="Hello")

        assert isinstance(response, ChatCompletionResponse)

    def test_call_with_single_messages(self) -> None:
        """Test __call__ with single message thread dispatches to forward."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Messages response"]]])
        response = mock_lm(messages=[{"role": "user", "content": "Hello"}])

        assert isinstance(response, ChatCompletionResponse)

    def test_call_with_batch_messages(self) -> None:
        """Test __call__ with batch messages dispatches to batch."""
        # Format: 1 layer, 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Response 1"], ["Response 2"]],
        ])
        responses = mock_lm(messages=[
            [{"role": "user", "content": "Hello"}],
            [{"role": "user", "content": "Hi"}],
        ])

        assert isinstance(responses, list)
        assert len(responses) == 2


class TestGenerativeLocalVLLMUsage:
    """Test usage statistics building using MockGenerativeLocalVLLM."""

    def test_response_has_usage(self) -> None:
        """Test that responses include usage statistics."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test response"]]])
        response = mock_lm.forward(prompt="Hello")

        assert isinstance(response.usage, Usage)
        # MockGenerativeLocalVLLM calculates usage based on word count
        assert response.usage.completion_tokens >= 0
        assert response.usage.total_tokens >= 0


class TestGenerativeLocalVLLMChoices:
    """Test choice building using MockGenerativeLocalVLLM."""

    def test_multiple_choices(self) -> None:
        """Test that multiple choices are correctly built."""
        # Format: 1 layer, 1 request, 3 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Choice 1", "Choice 2", "Choice 3"]]])
        response = mock_lm.forward(
            prompt="Hello",
            sampling_params=SamplingParams(n=3),
        )

        assert len(response.choices) == 3
        for i, choice in enumerate(response.choices):
            assert isinstance(choice, Choice)
            assert choice.index == i
            assert choice.message.content == f"Choice {i + 1}"


class TestGenerativeLocalVLLMModelName:
    """Test model_name property using MockGenerativeLocalVLLM."""

    def test_model_name_property(self) -> None:
        """Test model_name property returns expected value."""
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test"]]])
        assert mock_lm.model_name == "mock-generative-model"


class TestGenerativeLocalVLLMValidation:
    """Test input validation using the real GenerativeLocalVLLM validation methods."""

    @pytest.fixture
    def mock_lm(self) -> MockGenerativeLocalVLLM:
        """Create a MockGenerativeLocalVLLM instance.

        Returns:
            A MockGenerativeLocalVLLM instance.
        """
        return MockGenerativeLocalVLLM(responses=[[["Test"]]])

    @pytest.mark.parametrize(
        # Parameter names
        [
            "prompt",
            "messages",
            "expected_exception",
            "expected_message",
        ],
        # Parameter values
        [
            pytest.param(
                "Hello",                            # prompt
                None,                               # messages
                None,                               # expected_exception
                None,                               # expected_message
                id="valid_prompt_only",
            ),
            pytest.param(
                None,                               # prompt
                [{"role": "user", "content": "Hello"}],  # messages
                None,                               # expected_exception
                None,                               # expected_message
                id="valid_messages_only",
            ),
            pytest.param(
                None,                               # prompt
                None,                               # messages
                ValueError,                         # expected_exception
                "Must provide either prompt or messages",  # expected_message
                id="invalid_neither_prompt_nor_messages",
            ),
        ],
    )
    def test_forward_validation(
        self,
        mock_lm: MockGenerativeLocalVLLM,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        expected_exception: type[Exception] | None,
        expected_message: str | None,
    ) -> None:
        """Test forward method validates inputs correctly.

        Args:
            mock_lm: MockGenerativeLocalVLLM instance.
            prompt: Prompt string to test.
            messages: Messages to test.
            expected_exception: Expected exception type.
            expected_message: Expected exception message substring.
        """
        if expected_exception is not None:
            with pytest.raises(expected_exception, match=expected_message):
                mock_lm.forward(prompt=prompt, messages=messages)
        else:
            response = mock_lm.forward(prompt=prompt, messages=messages)
            assert isinstance(response, ChatCompletionResponse)


class TestGenerativeLocalVLLMMultipleOutputs:
    """Test multiple outputs (n > 1) using MockGenerativeLocalVLLM."""

    def test_multiple_outputs_single_call(self) -> None:
        """Test generating multiple outputs in a single call."""
        # Format: 1 layer, 1 request, 3 completions
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Output 1", "Output 2", "Output 3"]],
        ])
        response = mock_lm.forward(
            prompt="Generate three outputs",
            sampling_params=SamplingParams(n=3),
        )

        assert len(response.choices) == 3
        assert response.choices[0].message.content == "Output 1"
        assert response.choices[1].message.content == "Output 2"
        assert response.choices[2].message.content == "Output 3"

    def test_batch_with_different_n_values(self) -> None:
        """Test batch processing with different n values per request."""
        # Format: 1 layer, 2 requests with different number of completions
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Single output"], ["Output A", "Output B"]],
        ])
        sampling_params = [
            SamplingParams(n=1),
            SamplingParams(n=2),
        ]
        responses = mock_lm.batch(
            prompts=["First", "Second"],
            sampling_params=sampling_params,
        )

        assert len(responses) == 2
        assert len(responses[0].choices) == 1
        assert len(responses[1].choices) == 2


class TestGenerativeLocalVLLMLayeredResponses:
    """Test layered responses for sequential calls using MockGenerativeLocalVLLM."""

    def test_layered_responses(self) -> None:
        """Test that mock returns different responses for sequential calls."""
        # Format: 2 layers, each with 1 request, 1 completion
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["First call response"]],
            [["Second call response"]],
        ])

        # First call
        response1 = mock_lm.forward(prompt="Call 1")
        assert response1.choices[0].message.content == "First call response"

        # Second call
        response2 = mock_lm.forward(prompt="Call 2")
        assert response2.choices[0].message.content == "Second call response"

    def test_layered_batch_responses(self) -> None:
        """Test layered responses for batch calls."""
        # Format: 2 layers, each with 2 requests, 1 completion each
        mock_lm = MockGenerativeLocalVLLM(responses=[
            [["Batch 1 Response 1"], ["Batch 1 Response 2"]],
            [["Batch 2 Response 1"], ["Batch 2 Response 2"]],
        ])

        # First batch call
        responses1 = mock_lm.batch(prompts=["A", "B"])
        assert responses1[0].choices[0].message.content == "Batch 1 Response 1"
        assert responses1[1].choices[0].message.content == "Batch 1 Response 2"

        # Second batch call
        responses2 = mock_lm.batch(prompts=["C", "D"])
        assert responses2[0].choices[0].message.content == "Batch 2 Response 1"
        assert responses2[1].choices[0].message.content == "Batch 2 Response 2"


class TestGenerativeLocalVLLMResponseReset:
    """Test response reset functionality using MockGenerativeLocalVLLM."""

    def test_set_responses(self) -> None:
        """Test setting new responses."""
        # Format: layers[requests[completions]]
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Initial response"]]])
        response1 = mock_lm.forward(prompt="Test")
        assert response1.choices[0].message.content == "Initial response"

        # Set new responses
        mock_lm.set_responses([[["New response"]]])
        response2 = mock_lm.forward(prompt="Test")
        assert response2.choices[0].message.content == "New response"

    def test_reset_responses(self) -> None:
        """Test resetting responses to empty."""
        mock_lm = MockGenerativeLocalVLLM(responses=[[["Test"]]])
        mock_lm.reset_responses()

        with pytest.raises(AssertionError, match="No chat responses"):
            mock_lm.forward(prompt="Test")


# =============================================================================
# Integration Tests (GPU Required)
# =============================================================================

# GPU Skip Marker
pytestmark_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU tests require GPU access",
)


@pytestmark_gpu
class TestGenerativeLocalVLLMIntegration:
    """Integration tests for GenerativeLocalVLLM using real models (requires GPU).

    These tests instantiate actual models and verify that generation works correctly
    with reasonable inputs. They verify that token counts in Usage objects are > 0.
    """

    @pytest.fixture(scope="class")
    def shared_gpu_model(self) -> Generator[GenerativeLocalVLLM, None, None]:
        """Shared GenerativeLocalVLLM fixture for all GPU integration tests.

        This fixture loads a model once and shares it across all GPU test methods
        to avoid loading multiple models and running out of GPU memory.

        Yields:
            GenerativeLocalVLLM: A real GenerativeLocalVLLM instance.
        """
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        base_path = "/projects/BSTEWART/model_storage"
        model_name = "Qwen3-4B-Instruct-2507"
        model_path = os.path.join(base_path, model_name)

        lm = None
        try:
            logger.info(f"Initializing shared GPU model from: {model_path}")
            lm = GenerativeLocalVLLM(
                model=model_path,
                tensor_parallel_size=1,
                dtype="auto",
                gpu_memory_utilization=0.9,
                max_model_len=4096,
                enforce_eager=True,
                verbosity=Verbosity.INFO,
            )
            logger.info("Shared GPU model initialized successfully")
            yield lm
        finally:
            # Cleanup after all GPU tests complete
            if lm is not None:
                logger.info("Cleaning up shared GPU model...")
                lm.kill()

    @pytest.mark.parametrize(
        # Parameter names
        [
            "test_name",
            "prompt",
            "messages",
            "max_tokens",
        ],
        # Parameter values
        [
            pytest.param(
                "simple_prompt",                    # test_name
                "What is 2+2?",                     # prompt
                None,                               # messages
                50,                                 # max_tokens
                id="simple_prompt_generation",
            ),
            pytest.param(
                "simple_messages",                  # test_name
                None,                               # prompt
                [{"role": "user", "content": "What is the capital of France?"}],  # messages
                50,                                 # max_tokens
                id="simple_messages_generation",
            ),
            pytest.param(
                "multi_turn_conversation",          # test_name
                None,                               # prompt
                [                                   # messages
                    {"role": "user", "content": "Hello!"},
                    {"role": "assistant", "content": "Hi there! How can I help you?"},
                    {"role": "user", "content": "What is Python?"},
                ],
                100,                                # max_tokens
                id="multi_turn_conversation",
            ),
        ],
    )
    def test_forward_integration(
        self,
        shared_gpu_model: GenerativeLocalVLLM,
        test_name: str,
        prompt: str | None,
        messages: list[dict[str, str]] | None,
        max_tokens: int,
    ) -> None:
        """Test forward method with real model generates valid responses.

        This test verifies that:
        1. The model generates a valid ChatCompletionResponse
        2. The response contains at least one choice
        3. The usage statistics have prompt_tokens > 0
        4. The usage statistics have completion_tokens > 0

        Args:
            shared_gpu_model: Real GenerativeLocalVLLM instance.
            test_name: Name of the test case.
            prompt: Prompt string to test.
            messages: Messages to test.
            max_tokens: Maximum tokens to generate.
        """
        sampling_params = SamplingParams(temperature=0.1, max_tokens=max_tokens)

        response = shared_gpu_model.forward(
            prompt=prompt,
            messages=messages,
            sampling_params=sampling_params,
        )

        # Verify response structure
        assert isinstance(response, ChatCompletionResponse), (
            f"Expected ChatCompletionResponse, got {type(response)}"
        )
        assert len(response.choices) >= 1, "Expected at least one choice"
        assert response.choices[0].message.content is not None, (
            "Expected non-None content in response"
        )
        assert len(response.choices[0].message.content) > 0, (
            "Expected non-empty content in response"
        )

        # Verify usage statistics
        assert response.usage.prompt_tokens > 0, (
            f"Expected prompt_tokens > 0, got {response.usage.prompt_tokens}"
        )
        assert response.usage.completion_tokens > 0, (
            f"Expected completion_tokens > 0, got {response.usage.completion_tokens}"
        )
        assert response.usage.total_tokens > 0, (
            f"Expected total_tokens > 0, got {response.usage.total_tokens}"
        )
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        ), "total_tokens should equal prompt_tokens + completion_tokens"

        logger.info(
            f"Test '{test_name}' passed:\n"
            f"\tprompt_tokens={response.usage.prompt_tokens}\n"
            f"\tcompletion_tokens={response.usage.completion_tokens}\n"
            f"\ttotal_tokens={response.usage.total_tokens}"
        )

    def test_batch_integration(self, shared_gpu_model: GenerativeLocalVLLM) -> None:
        """Test batch method with real model generates valid responses.

        This test verifies that:
        1. The model generates a list of ChatCompletionResponse objects
        2. Each response contains valid usage statistics with tokens > 0

        Args:
            shared_gpu_model: Real GenerativeLocalVLLM instance.
        """
        prompts = [
            "What is 2+2?",
            "What is the capital of France?",
            "Explain gravity in one sentence.",
        ]
        sampling_params = SamplingParams(temperature=0.1, max_tokens=50)

        responses = shared_gpu_model.batch(
            prompts=prompts,
            sampling_params=sampling_params,
        )

        # Verify response structure
        assert isinstance(responses, list), f"Expected list, got {type(responses)}"
        assert len(responses) == len(prompts), (
            f"Expected {len(prompts)} responses, got {len(responses)}"
        )

        for i, response in enumerate(responses):
            assert isinstance(response, ChatCompletionResponse), (
                f"Response {i}: Expected ChatCompletionResponse, got {type(response)}"
            )
            assert len(response.choices) >= 1, f"Response {i}: Expected at least one choice"

            # Verify usage statistics
            assert response.usage.prompt_tokens > 0, (
                f"Response {i}: Expected prompt_tokens > 0, got {response.usage.prompt_tokens}"
            )
            assert response.usage.completion_tokens > 0, (
                f"Response {i}: Expected completion_tokens > 0, "
                f"got {response.usage.completion_tokens}"
            )
            assert response.usage.total_tokens > 0, (
                f"Response {i}: Expected total_tokens > 0, got {response.usage.total_tokens}"
            )

        logger.info(f"Batch test passed with {len(responses)} responses")

    def test_multiple_completions_integration(
        self, shared_gpu_model: GenerativeLocalVLLM
    ) -> None:
        """Test generation with n > 1 produces multiple completions.

        This test verifies that:
        1. When n > 1, the model generates multiple choices
        2. Each choice has valid content
        3. Usage statistics reflect all completions

        Args:
            shared_gpu_model: Real GenerativeLocalVLLM instance.
        """
        sampling_params = SamplingParams(temperature=0.7, max_tokens=50, n=3)

        response = shared_gpu_model.forward(
            prompt="Give me a creative name for a cat.",
            sampling_params=sampling_params,
        )

        # Verify multiple choices
        assert len(response.choices) == 3, f"Expected 3 choices, got {len(response.choices)}"

        for i, choice in enumerate(response.choices):
            assert choice.message.content is not None, f"Choice {i}: Expected non-None content"
            assert len(choice.message.content) > 0, f"Choice {i}: Expected non-empty content"

        # Verify usage statistics
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0

        logger.info(
            f"Multiple completions test passed:\n"
            f"\tnum_choices={len(response.choices)}\n"
            f"\tcompletion_tokens={response.usage.completion_tokens}\n"
            f"\ttotal_tokens={response.usage.total_tokens}"
        )


if __name__ == "__main__":
    gpu_available = torch.cuda.is_available()
    if not gpu_available:
        # Run all tests except GPU-specific ones
        pytest.main([__file__, "-vv"])
    else:
        # If GPU is available, run all tests including GPU-specific ones
        pytest.main([
            __file__,
            "-v",
            "-s",
            "--log-cli-level=INFO",
        ])
