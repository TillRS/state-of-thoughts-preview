"""
Tests for the utility functions in test_utils.py.
"""

from typing import Any, Optional

import pytest

from misc_utils import (
	_is_instance_of_type,
	_is_optional_type,
	format_list_of_fields,
	parse_base_signature,
)


@pytest.mark.parametrize(
	"input_fields, expected_output",
	[
		(["field1", "field2"], "`field1` and `field2`"),
		(["field1"], "`field1`"),
		(["a", "b", "c", "d"], "`a`, `b`, `c`, and `d`"),
	],
)
def test_format_list_of_fields(input_fields: list[str], expected_output: str) -> None:
	"""
	Test the format_list_of_fields function with various input cases.
	"""
	assert format_list_of_fields(input_fields) == expected_output


def test_format_list_of_fields_empty() -> None:
	"""
	Test the format_list_of_fields function with an empty list.
	This should raise an AssertionError.
	"""
	with pytest.raises(AssertionError):
		format_list_of_fields([])


@pytest.mark.parametrize(
	"input_field_names, output_field_names, expected_output",
	[
		(
			["input1", "input2"],
			["output1"],
			("`input1` and `input2`", "`output1`"),
		),
		(
			["input1"],
			["output1", "output2"],
			("`input1`", "`output1` and `output2`"),
		),
		(
			["input1", "input2", "input3"],
			["output1", "output2"],
			("`input1`, `input2`, and `input3`", "`output1` and `output2`"),
		),
		(["input1"], ["output1"], ("`input1`", "`output1`")),
		(
			["input1", "input2"],
			["output1", "output2"],
			("`input1` and `input2`", "`output1` and `output2`"),
		),
	],
)
def test_parse_base_signature(
	input_field_names: list[str],
	output_field_names: list[str],
	expected_output: tuple[str, str],
) -> None:
	"""
	Test the parse_base_signature function with various input and output field names.
	"""
	assert (
		parse_base_signature(input_field_names, output_field_names) == expected_output
	)


@pytest.mark.parametrize(
	"value, type_hint, expected",
	[
		# Simple types - positive cases
		("hello", str, True),
		(42, int, True),
		(3.14, float, True),
		(True, bool, True),
		# Simple types - negative cases
		("hello", int, False),
		(42, str, False),
		# List types - positive cases
		(["a", "b"], list[str], True),
		([1, 2, 3], list[int], True),
		([], list[str], True),
		# List types - negative cases
		([1, 2], list[str], False),
		(["a", "b"], list[int], False),
		("not a list", list[str], False),
		# Nested list types - positive cases
		([["a", "b"], ["c"]], list[list[str]], True),
		([[1, 2], [3, 4]], list[list[int]], True),
		([], list[list[str]], True),
		# Nested list types - negative cases
		([["a"], [1]], list[list[str]], False),
		(["a", "b"], list[list[str]], False),
		# Dict types - positive cases
		({"a": 1, "b": 2}, dict[str, int], True),
		({}, dict[str, int], True),
		# Dict types - negative cases
		({1: "a"}, dict[str, int], False),
		({"a": "b"}, dict[str, int], False),
		# Union types - positive cases
		("hello", str | int, True),
		(42, str | int, True),
		# Union types - negative cases
		(3.14, str | int, False),
		# Optional types - positive cases (Optional[str] is Union[str, None])
		("hello", Optional[str], True),
		(None, Optional[str], True),
		# Optional types - negative cases
		(42, Optional[str], False),
	],
)
def test_is_instance_of_type(value: Any, type_hint: Any, expected: bool) -> None:
	"""Test the _is_instance_of_type helper function with various type combinations."""
	assert _is_instance_of_type(value, type_hint) is expected


@pytest.mark.parametrize(
	"type_hint, expected",
	[
		# Non-optional types - should return False
		(str, False),
		(int, False),
		(list[str], False),
		(dict[str, int], False),
		(list[list[str]], False),
		# Optional types using Optional[T] syntax - should return True
		(Optional[str], True),
		(Optional[int], True),
		(Optional[list[str]], True),
		# Optional types using T | None syntax (Python 3.10+) - should return True
		(str | None, True),
		(int | None, True),
		(list[str] | None, True),
		(dict[str, int] | None, True),
		# Union types without None - should return False
		(str | int, False),
		(str | int | float, False),
	],
)
def test_is_optional_type(type_hint: Any, expected: bool) -> None:
	"""Test the _is_optional_type helper function with various type hints."""
	assert _is_optional_type(type_hint) is expected


if __name__ == "__main__":
	# Run the tests
	pytest.main([__file__, "-vv"])
