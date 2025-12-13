"""
Tree of Thoughts Parameters

This module provides parameter classes for configuring Tree of Thoughts search behavior.
"""

from dataclasses import dataclass
from enum import StrEnum

from constants import CandidateGenerationMethod
from lm import DEFAULT_TEMPERATURE


class NodeSelectionStrategy(StrEnum):
	"""Strategy for selecting nodes during tree search."""

	GREEDY = "greedy"
	SAMPLE = "sample"


@dataclass
class TreeOfThoughtsParameters:
	"""
	Parameters for controlling Tree of Thoughts search behavior.

	This class replaces the old utils.search_parameters.TreeOfThoughtsParameters
	with a simplified, modern implementation focused on local VLLM models.
	"""

	# Tree search parameters
	depth: int = 3
	# Maximum depth of the reasoning tree (number of reasoning steps).

	n_samples_generation: int = 3
	# Number of candidate reasoning steps to generate at each node.

	n_samples_judge: int = 1
	# Number of evaluation samples to use when scoring candidates.

	n_final_responses_per_trajectory: int | None = None
	# Number of final responses to generate per frontier node in the final layer.
	# If None, falls back to n_samples_generation.

	top_k: int = 2
	# Number of top-scoring candidates to keep at each layer.

	num_final_candidates: int = 1
	# Number of final candidates to return from the tree search.
	# TODO[P2]: We should include an easy option to return all final candidates.

	# Temperature parameters
	generation_temperature: float = DEFAULT_TEMPERATURE
	# Temperature for generating reasoning step candidates.

	judge_temperature: float = 0.0
	# Temperature for evaluating candidate quality.

	# Search behavior
	do_pruning: bool = True
	# Whether to prune low-scoring candidates during search.

	do_early_stopping: bool = False
	# Whether to allow early stopping based on controller decisions.

	use_self_consistency: bool = False
	# Whether to use self-consistency for final answer selection.

	candidate_generation_method: CandidateGenerationMethod = (
		CandidateGenerationMethod.SINGLE_CANDIDATE_CALLS
	)
	# Method for generating multiple candidates (single calls vs multi-candidate).

	node_selection_strategy: NodeSelectionStrategy = NodeSelectionStrategy.GREEDY
	# Strategy for selecting which nodes to expand.

	# Controller-specific sampling parameters
	controller_temperature: float | None = None
	# Temperature override specifically for controller. If None, uses generation_temperature.

	controller_top_p: float | None = None
	# Cumulative probability for controller nucleus sampling (0-1]. If None, uses model default.

	controller_top_k: int | None = None
	# Number of top tokens for controller to consider. Set to -1 for all tokens. If None, uses model default.

	controller_min_p: float | None = None
	# Minimum probability for controller tokens, relative to most likely token [0-1]. If None, uses model default.

	controller_use_beam_search: bool = False
	# Whether to use beam search instead of sampling for controller decisions.

	# Generator-specific sampling parameters
	generator_temperature: float | None = None
	# Temperature override specifically for generator. If None, uses generation_temperature.

	generator_top_p: float | None = None
	# Cumulative probability for generator nucleus sampling (0-1]. If None, uses model default.

	generator_top_k: int | None = None
	# Number of top tokens for generator to consider. Set to -1 for all tokens. If None, uses model default.

	generator_min_p: float | None = None
	# Minimum probability for generator tokens, relative to most likely token [0-1]. If None, uses model default.

	generator_use_beam_search: bool = False
	# Whether to use beam search instead of sampling for generator decisions.

	def __post_init__(self) -> None:
		"""Validate parameter values after initialization."""
		assert self.depth > 0, "depth must be greater than 0"
		assert self.n_samples_generation > 0, (
			"n_samples_generation must be greater than 0"
		)
		if self.n_final_responses_per_trajectory is not None:
			assert self.n_final_responses_per_trajectory > 0, (
				"n_final_responses_per_trajectory must be greater than 0 when provided"
			)
		assert self.n_samples_judge > 0, "n_samples_judge must be greater than 0"
		assert self.top_k > 0, "top_k must be greater than 0"
		assert self.num_final_candidates > 0, (
			"num_final_candidates must be greater than 0"
		)
		assert 0 <= self.generation_temperature <= 2, (
			"generation_temperature must be between 0 and 2"
		)
		assert 0 <= self.judge_temperature <= 2, (
			"judge_temperature must be between 0 and 2"
		)

		if self.do_pruning:
			assert (
				self.num_final_candidates <= self.top_k * self.n_samples_generation
			), (
				f"num_final_candidates ({self.num_final_candidates}) must be <= top_k * n_samples_generation ({self.top_k * self.n_samples_generation})"
			)

		assert self.n_samples_generation >= self.top_k, (
			"n_samples_generation must be >= top_k"
		)

		# Validate controller-specific parameters
		if self.controller_temperature is not None:
			assert 0 <= self.controller_temperature <= 2, (
				"controller_temperature must be between 0 and 2"
			)

		if self.controller_top_p is not None:
			assert 0 < self.controller_top_p <= 1, (
				"controller_top_p must be in (0, 1]"
			)

		if self.controller_min_p is not None:
			assert 0 <= self.controller_min_p <= 1, (
				"controller_min_p must be in [0, 1]"
			)

		# Validate generator-specific parameters
		if self.generator_temperature is not None:
			assert 0 <= self.generator_temperature <= 2, (
				"generator_temperature must be between 0 and 2"
			)

		if self.generator_top_p is not None:
			assert 0 < self.generator_top_p <= 1, (
				"generator_top_p must be in (0, 1]"
			)

		if self.generator_min_p is not None:
			assert 0 <= self.generator_min_p <= 1, (
				"generator_min_p must be in [0, 1]"
			)
