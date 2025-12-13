"""Utilities for testing DSPy reasoning components.

This module provides mock classes for unit testing with mock vLLM LLM objects,
mock tokenizers, mock LocalPredict (dspy.LM) objects, and mock Predict objects.

The mocks use real vLLM output types (RequestOutput, ScoringRequestOutput) directly.

Response formats (no normalization - tests must provide exact format):
    - Chat responses: list[list[list[str]]]. Shape:
		[num_layers, num_input_messages, num_choices_per_input_message]
    - Score responses: list[list[float]]. Shape:
		[num_layers, num_query_document_pairs]
"""

# Standard library imports
import logging
import uuid
from typing import Any

# Third-party imports
import dspy
from vllm import RequestOutput, SamplingParams, ScoringRequestOutput
from vllm.outputs import CompletionOutput, ScoringOutput

# Local imports
from lm.generative_local_lm import GenerativeLocalVLLM
from lm.lm_constants import FinishReason, SamplingParam, TaskType
from lm.scoring_local_lm import ScoringLocalVLLM
from predict.local_predict import LocalPredict

logger = logging.getLogger(__name__)


# =============================================================================
# Mock Tokenizer
# =============================================================================


class MockTokenizer:
    """Mock tokenizer that simulates token counting by word splitting."""

    def __call__(self, text: str) -> list[str]:
        """Tokenize text by splitting on whitespace (i.e., splitting whole words)."""
        if not text:
            return []
        return text.split()


# =============================================================================
# Mock vLLM LLM Class
# =============================================================================


class MockVLLM:
    """Mock for vllm.LLM that returns pre-programmed responses.

    This mock simulates the vLLM LLM class behavior without loading actual models.
    It supports layered responses for sequential calls and returns real vLLM output types.

    Response formats (strict - no normalization):
        - chat_responses: list[list[list[str]]] = layers[requests[completions]]
        - score_responses: list[list[float]] = layers[scores]
    """

    def __init__(
        self,
        chat_responses: list[list[list[str]]] | None = None,
        score_responses: list[list[float]] | None = None,
    ) -> None:
        """Initialize MockVLLM with responses.

        Args:
            chat_responses: Layered chat responses.
                Format: layers[requests[completions[text]]]
            score_responses: Layered score responses.
                Format: layers[scores[float]]
        """
        self._chat_responses = chat_responses or []
        self._score_responses = score_responses or []
        self._chat_layer_index = 0
        self._score_layer_index = 0
        self._tokenizer = MockTokenizer()

    def get_tokenizer(self) -> MockTokenizer:
        """Get the mock tokenizer."""
        return self._tokenizer

    def chat(
        self,
        messages: list[list[dict[str, str]]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[RequestOutput]:
        """Mock the chat method of vLLM.

        Returns real vLLM RequestOutput objects.
        """
        assert self._chat_responses, "No chat responses provided for MockVLLM"
        assert self._chat_layer_index < len(self._chat_responses), (
            f"Not enough chat response layers. "
            f"Requested layer {self._chat_layer_index}, "
            f"but only {len(self._chat_responses)} available."
        )

        layer = self._chat_responses[self._chat_layer_index]
        batch_size = len(messages)

        assert batch_size <= len(layer), (
            f"Not enough responses in chat layer {self._chat_layer_index}. "
            f"Requested {batch_size} responses, but only {len(layer)} available."
        )

        # Build real RequestOutput objects
        results: list[RequestOutput] = []
        for i in range(batch_size):
            completions = layer[i]
            outputs = [
                CompletionOutput(
                    index=j,
                    text=text,
                    token_ids=[],
                    cumulative_logprob=None,
                    logprobs=None,
                    finish_reason=FinishReason.STOP,
                )
                for j, text in enumerate(completions)
            ]
            results.append(RequestOutput(
                request_id=f"req_{uuid.uuid4()}",
                prompt=None,
                prompt_token_ids=[],
                prompt_logprobs=None,
                outputs=outputs,
                finished=True,
            ))

        self._chat_layer_index += 1
        return results

    def score(
        self,
        queries: list[str],
        documents: list[str],
        use_tqdm: bool = True,
        **kwargs: Any,
    ) -> list[ScoringRequestOutput]:
        """Mock the score method of vLLM.

        Returns real vLLM ScoringRequestOutput objects.
        """
        assert self._score_responses, "No score responses provided for MockVLLM"
        assert self._score_layer_index < len(self._score_responses), (
            f"Not enough score response layers. "
            f"Requested layer {self._score_layer_index}, "
            f"but only {len(self._score_responses)} available."
        )

        layer = self._score_responses[self._score_layer_index]
        num_pairs = len(queries)

        assert num_pairs <= len(layer), (
            f"Not enough scores in layer {self._score_layer_index}. "
            f"Requested {num_pairs} scores, but only {len(layer)} available."
        )

        # Build real ScoringRequestOutput objects
        results = [
            ScoringRequestOutput(
                request_id=f"score_{uuid.uuid4()}",
                outputs=ScoringOutput(score=layer[i]),
                prompt_token_ids=[],
                finished=True,
                num_cached_tokens=0,
            )
            for i in range(num_pairs)
        ]
        self._score_layer_index += 1
        return results

    def set_chat_responses(self, responses: list[list[list[str]]]) -> None:
        """Set new chat responses and reset layer index."""
        self._chat_responses = responses
        self._chat_layer_index = 0

    def set_score_responses(self, responses: list[list[float]]) -> None:
        """Set new score responses and reset layer index."""
        self._score_responses = responses
        self._score_layer_index = 0


# =============================================================================
# Mock Language Model Classes
# =============================================================================


class MockGenerativeLocalVLLM(GenerativeLocalVLLM):
    """Mock of GenerativeLocalVLLM that injects MockVLLM instead of real vLLM.

    This mock tests the actual class logic while only mocking the vLLM layer.

    Response format:
        list[list[list[str]]]. Shape:
            [num_layers, num_input_messages, num_choices_per_input_message]
    """

    def __init__(
        self,
        responses: list[list[list[str]]] | None = None,
    ) -> None:
        """Initialize the mock with responses for generation tasks.

        Args:
            responses: Chat responses in vLLM format. Shape:
                [num_layers, num_input_messages, num_choices_per_input_message]
        """
        # Skip parent __init__ to avoid heavy vLLM initialization
        self.history: list[dict[str, Any]] = []
        self.model_path = "mock-generative-model"
        self._model_name = "mock-generative-model"
        self.model_type = "chat"
        self._verbosity = None

        self.kwargs = {
            SamplingParam.TEMPERATURE: 0.0,
            SamplingParam.N: 1,
        }

        self.model = MockVLLM(chat_responses=responses)
        self.tokenizer = self.model.get_tokenizer()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def responses(self) -> list[list[list[str]]]:
        """Get the current responses in vLLM format."""
        return self.model._chat_responses

    def set_responses(self, responses: list[list[list[str]]] | None = None) -> None:
        """Set responses for the mock in vLLM format. Shape:
            [num_layers, num_input_messages, num_choices_per_input_message].

        Args:
            responses: Chat responses in vLLM format. Shape:
                [num_layers, num_input_messages, num_choices_per_input_message].
        """
        self.model.set_chat_responses(responses if responses is not None else [])

    def reset_responses(self) -> None:
        """Reset responses to empty list."""
        self.model.set_chat_responses([])

    def kill(self) -> None:
        """No-op for mock."""
        pass


class MockScoringLocalVLLM(ScoringLocalVLLM):
    """Mock of ScoringLocalVLLM that injects MockVLLM instead of real vLLM.

    This mock tests the actual class logic while only mocking the vLLM layer.
    """

    def __init__(
        self,
        rerank_responses: list[list[float]] | None = None,
    ) -> None:
        """Initialize the mock with rerank responses for scoring tasks.

        Args:
            rerank_responses: Score responses in vLLM format.
                Format: list[list[float]] = layers[scores]
        """
        # Skip parent __init__ to avoid heavy vLLM initialization
        self.history: list[dict[str, Any]] = []
        self.model_path = "mock-scoring-model"
        self._model_name = "mock-scoring-model"
        self._verbosity = None
        self.kwargs: dict[str, Any] = {}
        self.task = TaskType.SCORE

        self.model = MockVLLM(score_responses=rerank_responses)
        self.tokenizer = self.model.get_tokenizer()

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def rerank_responses(self) -> list[list[float]]:
        """Get the current rerank responses in vLLM format."""
        return self.model._score_responses

    def set_rerank_responses(self, rerank_responses: list[list[float]] | None = None) -> None:
        """Set rerank responses for the mock."""
        self.model.set_score_responses(rerank_responses or [])

    def reset_rerank_responses(self) -> None:
        """Reset rerank responses to empty list."""
        self.set_rerank_responses()

    def kill(self) -> None:
        """No-op for mock."""
        pass


class MockPredict(LocalPredict):
    """Mock LocalPredict for testing."""

    def __init__(
        self,
        responses: list[list[list[str]]],
        signature: dspy.Signature,
    ) -> None:
        """Initialize MockPredict.

        Args:
            responses: Chat responses in vLLM format.
                Format: list[list[list[str]]] = layers[requests[completions]]
            signature: DSPy signature for the prediction.
        """
        mock_lm = MockGenerativeLocalVLLM(responses=responses)
        super().__init__(signature=signature, lm=mock_lm)
        self.lm = mock_lm
        self._responses = responses
        self._signature = signature
