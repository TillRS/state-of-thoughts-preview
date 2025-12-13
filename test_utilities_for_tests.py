"""Tests for utilities_for_tests.py mock classes.

Tests the mock implementations that simulate vLLM behavior for unit testing.

Response formats (strict):
    - Chat responses: list[list[list[str]]] = layers[requests[completions]]
    - Score responses: list[list[float]] = layers[scores]
"""

# Third-party imports
import dspy
import pytest
from vllm import SamplingParams

# Local imports
from lm.lm_constants import MessageKey, MessageRole, TaskType
from utilities_for_tests import (
    MockGenerativeLocalVLLM,
    MockPredict,
    MockScoringLocalVLLM,
)

# =============================================================================
# Tests for MockGenerativeLocalVLLM
# =============================================================================


class TestMockGenerativeLocalVLLM:
    """Tests for MockGenerativeLocalVLLM generation functionality."""

    def test_init(self) -> None:
        """Test MockGenerativeLocalVLLM initializes correctly."""
        # Format: layers[requests[completions]]
        responses = [[["test"]]]
        mock = MockGenerativeLocalVLLM(responses=responses)
        assert mock.responses == responses

    def test_init_empty(self) -> None:
        """Test MockGenerativeLocalVLLM initializes with no responses."""
        mock = MockGenerativeLocalVLLM()
        assert mock.responses == []

    def test_set_responses(self) -> None:
        """Test MockGenerativeLocalVLLM.set_responses updates responses."""
        mock = MockGenerativeLocalVLLM()
        new_responses = [[["new"]]]
        mock.set_responses(new_responses)
        assert mock.responses == new_responses

    def test_reset(self) -> None:
        """Test MockGenerativeLocalVLLM.reset_responses clears responses."""
        mock = MockGenerativeLocalVLLM(responses=[[["test"]]])
        mock.reset_responses()
        assert mock.responses == []

    def test_forward(self) -> None:
        """Test MockGenerativeLocalVLLM.forward uses real class logic."""
        # Format: layers[requests[completions]] - 2 layers for 2 calls
        mock = MockGenerativeLocalVLLM(responses=[
            [["response_1"]],  # Layer 0: 1 request, 1 completion
            [["response_2"]],  # Layer 1: 1 request, 1 completion
        ])

        messages = [{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "Hello"}]
        sp = SamplingParams(n=1)

        response = mock.forward(messages=messages, sampling_params=sp)
        assert response.choices[0].message.content == "response_1"

    def test_batch(self) -> None:
        """Test MockGenerativeLocalVLLM.batch uses real class logic."""
        # Format: layers[requests[completions]] - 1 layer with 2 requests
        mock = MockGenerativeLocalVLLM(responses=[
            [["r1"], ["r2"]],  # Layer 0: 2 requests, 1 completion each
        ])

        messages = [
            [{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "1"}],
            [{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "2"}],
        ]
        sp = SamplingParams(n=1)

        responses = mock.batch(messages=messages, sampling_params=sp)
        assert len(responses) == 2
        assert responses[0].choices[0].message.content == "r1"
        assert responses[1].choices[0].message.content == "r2"

    def test_multiple_completions(self) -> None:
        """Test MockGenerativeLocalVLLM with multiple completions per request."""
        # Format: 1 layer, 1 request, 3 completions
        mock = MockGenerativeLocalVLLM(responses=[
            [["comp1", "comp2", "comp3"]],
        ])

        messages = [{MessageKey.ROLE: MessageRole.USER, MessageKey.CONTENT: "Hello"}]
        sp = SamplingParams(n=3)

        response = mock.forward(messages=messages, sampling_params=sp)
        assert len(response.choices) == 3
        assert response.choices[0].message.content == "comp1"
        assert response.choices[1].message.content == "comp2"
        assert response.choices[2].message.content == "comp3"


# =============================================================================
# Tests for MockScoringLocalVLLM
# =============================================================================


class TestMockScoringLocalVLLM:
    """Tests for MockScoringLocalVLLM scoring functionality."""

    def test_init(self) -> None:
        """Test MockScoringLocalVLLM initializes correctly."""
        # Format: layers[scores]
        responses = [[0.9, 0.1]]
        mock = MockScoringLocalVLLM(rerank_responses=responses)
        assert mock.task == TaskType.SCORE
        assert mock.rerank_responses == responses

    def test_init_empty(self) -> None:
        """Test MockScoringLocalVLLM initializes with no responses."""
        mock = MockScoringLocalVLLM()
        assert mock.rerank_responses == []

    def test_set_responses(self) -> None:
        """Test MockScoringLocalVLLM.set_rerank_responses updates responses."""
        mock = MockScoringLocalVLLM()
        new_responses = [[0.5]]
        mock.set_rerank_responses(new_responses)
        assert mock.rerank_responses == new_responses

    def test_reset(self) -> None:
        """Test MockScoringLocalVLLM.reset_rerank_responses clears responses."""
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9]])
        mock.reset_rerank_responses()
        assert mock.rerank_responses == []

    def test_pairwise_scoring(self) -> None:
        """Test MockScoringLocalVLLM for pairwise scoring."""
        # Format: layers[scores] - 1 layer with 2 scores
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9, 0.1]])

        queries = ["q1", "q2"]
        documents = ["d1", "d2"]

        responses = mock.batch(queries=queries, documents=documents, broadcast_scores=False)
        assert len(responses) == 2
        assert responses[0].results[0].relevance_score == 0.9
        assert responses[1].results[0].relevance_score == 0.1

    def test_broadcast_scoring(self) -> None:
        """Test MockScoringLocalVLLM for broadcast scoring."""
        # Format: layers[scores] - 1 layer with 2 scores for q1-d1, q1-d2
        mock = MockScoringLocalVLLM(rerank_responses=[[0.9, 0.1]])

        queries = ["q1", "q1"]
        documents = ["d1", "d2"]

        responses = mock.batch(queries=queries, documents=documents, broadcast_scores=True)
        assert len(responses) == 1
        assert len(responses[0].results) == 2

    def test_layered_responses(self) -> None:
        """Test MockScoringLocalVLLM with multiple layers for sequential calls."""
        # Format: 2 layers, each with 1 score
        mock = MockScoringLocalVLLM(rerank_responses=[
            [0.9],  # Layer 0
            [0.5],  # Layer 1
        ])

        response1 = mock.forward(query="Q1", document="D1")
        assert response1.results[0].relevance_score == 0.9

        response2 = mock.forward(query="Q2", document="D2")
        assert response2.results[0].relevance_score == 0.5


# =============================================================================
# Tests for MockPredict
# =============================================================================


class TestMockPredict:
    """Tests for MockPredict functionality."""

    def test_init(self) -> None:
        """Test MockPredict initializes with responses."""
        sig = dspy.Signature("input -> output")
        # Format: layers[requests[completions]]
        responses = [[["output"]]]
        mock = MockPredict(responses=responses, signature=sig)
        assert isinstance(mock.lm, MockGenerativeLocalVLLM)
        assert mock.lm.responses == responses

    def test_forward(self) -> None:
        """Test MockPredict.forward calls through to LocalPredict."""
        sig = dspy.Signature("input -> output")
        # Format: layers[requests[completions]]
        responses = [[["## output\nvalue"]]]
        mock = MockPredict(responses=responses, signature=sig)

        pred = mock.forward(input="test")
        assert len(pred) == 1
        assert pred[0].output == "value"

if __name__ == "__main__":
    pytest.main([__file__])
