# Standard library imports

# Third-party imports
from dspy.adapters.utils import get_annotation_name
from pydantic.fields import FieldInfo

# Local imports
from adapter.adapter_constants import FinalOutputKind
from adapter.prompts import (
	FINAL_OUTPUT_CONCLUSION,
	FINAL_OUTPUT_SYNTHESIS_FAITHFUL,
	FINAL_OUTPUT_SYNTHESIS_RESTRUCTURED,
	FINAL_OUTPUT_SYNTHESIS_STRICT,
)
from signatures import ReasoningSignature


def uncapitalize_first_letter(s: str) -> str:
	"""
	Uncapitalize the first letter of a string.

	Parameters:
	    s (str): The input string.

	Returns:
	    str: The input string with the first letter uncapitalized.
	"""
	if not s:
		return s
	return s[0].lower() + s[1:]


def get_field_description_string(fields: dict[str, FieldInfo]) -> str:
	"""
	Produce a string that describes a dictionary of either input or output fields.

	Args:
	    fields: A dictionary of field names to field information. This information includes the
	        name of the field (key in the dictionary), the type of the field (annotation of the
	        value), a description of the field (if available via the json schema of the field value),
	        and constraints (if available through the `json_schema_extra` dictionary of the field
	        value).

	Returns:
	    A string that describes the collection of input or output fields.
	"""
	field_descriptions = []
	for idx, (k, v) in enumerate(fields.items()):
		field_message = f"{idx + 1}. `{k}`" if len(fields) > 1 else f"`{k}`"
		field_message += f" ({get_annotation_name(v.annotation)})"

		# Handle desc if it exists
		if v.json_schema_extra and "desc" in v.json_schema_extra: # pyright: ignore[reportOperatorIssue]
			desc = v.json_schema_extra.get("desc")
			if desc and desc != f"${{{k}}}":
				field_message += f": {desc}"

		# Format constraints as bulletized list
		constraints = (
			v.json_schema_extra.get("constraints") if v.json_schema_extra else None
		)
		if constraints and isinstance(constraints, str):
			# First handle the case with two constraints connected by "and" without commas
			if " and " in constraints and "," not in constraints:
				individual_constraints = [s.strip() for s in constraints.split(" and ")]
			else:
				# Handle complex cases with commas and "and" (i.e., 3 or more constraints)
				individual_constraints = []
				for c in constraints.split(","):
					c = c.strip()
					if c:
						# Remove "and" prefix if it starts with "and "
						if c.startswith("and "):
							c = c[4:]
						individual_constraints.append(c)

			if individual_constraints:
				field_message += "\n\tConstraints:"
				for constraint in individual_constraints:
					field_message += f"\n\t\t* {constraint}"
		field_descriptions.append(field_message)
	return "\n".join(field_descriptions).strip()


def format_field_description(signature: type[ReasoningSignature]) -> str:
	"""Format the field description for the system message.

	This method formats the field description for the system message. It should return a string
	that contains the field description for the input fields and the output fields.

	Args:
	    signature: The DSPy signature for which to format the field description.

	Returns:
	    A string that contains the field description for the input fields and the output fields.
	"""
	if len(signature.input_fields) > 1:
		inputs_msg = "Your inputs will be:\n"
	else:
		inputs_msg = "Your input is:\n"
	inputs_msg += get_field_description_string(signature.input_fields)

	if len(signature.output_fields) > 1:
		outputs_msg = "Your goal is to produce the following outputs:\n"
	else:
		outputs_msg = "Your goal is to produce the following output:\n"
	outputs_msg += get_field_description_string(signature.output_fields)
	return f"{inputs_msg}\n\n{outputs_msg}"


def generate_output_field_sections(output_field_names: list[str]) -> str:
	"""
	Generate formatted sections for each output field.

	Args:
	    output_field_names: List of output field names

	Returns:
	    Formatted string with sections for each output field
	"""
	output_field_sections = ""
	for field in output_field_names:
		output_field_sections += f"## {field}\nYour response for `{field}` here\n"
	return output_field_sections.strip()


def get_final_output_description(final_output_kind: FinalOutputKind) -> str:
	"""
	Get the final output description string for the given kind.

	Parameters:
	    final_output_kind: The kind of final output instruction.

	Returns:
	    The corresponding description string.

	Raises:
	    ValueError: If the final_output_kind is not recognized.
	"""
	if final_output_kind == FinalOutputKind.SYNTHESIS_STRICT:
		return FINAL_OUTPUT_SYNTHESIS_STRICT
	elif final_output_kind == FinalOutputKind.SYNTHESIS_FAITHFUL:
		return FINAL_OUTPUT_SYNTHESIS_FAITHFUL
	elif final_output_kind == FinalOutputKind.SYNTHESIS_RESTRUCTURED:
		return FINAL_OUTPUT_SYNTHESIS_RESTRUCTURED
	elif final_output_kind == FinalOutputKind.CONCLUSION:
		return FINAL_OUTPUT_CONCLUSION
	else:
		raise ValueError(f"Unknown final_output_kind: {final_output_kind}")
