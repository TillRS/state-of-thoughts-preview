"""
Default demonstration examples for the TreeOfThoughtGenerator generator.

Provides smart default demos for reasoning step generation and final answer generation.
"""

# Standard library imports

# Third-party imports

# Local imports
from predict.controller_constants import ControllerOutputParameters
from signatures import ArgumentStance
from signatures.example_signatures import ArgumentField, MathField
from tree.tree_constants import ReasoningState

# Reasoning demos - for generating intermediate reasoning steps
REASONING_DEMOS = [
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 12 + 8?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I need to add two numbers together",
				MathField.MATH_OPERATION: "12 + 8 equals 20",
			}
		],
		ReasoningState.OUTPUT: {},
	},
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is the capital of France?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should recall my knowledge of European capitals",
				MathField.MATH_OPERATION: "France is a country in Western Europe, and its capital city is Paris",
			}
		],
		ReasoningState.OUTPUT: {},
	},
]

# Answer demos - for generating final answers after reasoning steps
ANSWER_DEMOS = [
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 5 + 3?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I'll add these numbers step by step",
				MathField.MATH_OPERATION: "5 + 3 equals 8",
			}
		],
		ReasoningState.OUTPUT: {
			MathField.ANSWER: "8",
		},
	},
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 2 * 6?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I need to multiply these numbers",
				MathField.MATH_OPERATION: "2 multiplied by 6 equals 12",
			}
		],
		ReasoningState.OUTPUT: {
			MathField.ANSWER: "12",
		},
	},
]
# The above are currently not used

MATH_DEMOS = [
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is (10 - 3) * 2?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I need to follow the order of operations",
				MathField.MATH_OPERATION: "First, I need to evaluate the expression in parentheses",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "Let me calculate the subtraction",
				MathField.MATH_OPERATION: "10 - 3 = 7",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "Now I can multiply the result",
				MathField.MATH_OPERATION: "Now multiply: 7 * 2 = 14",
			},
		],
		ReasoningState.OUTPUT: {
            MathField.ANSWER: "14"
        },
	},
	{
		ReasoningState.INPUT: {MathField.MATH_PROBLEM: "What is 15 + 8 - 5?"},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should evaluate from left to right",
				MathField.MATH_OPERATION: "I'll work left to right",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "Let me add the first two numbers",
				MathField.MATH_OPERATION: "15 + 8 = 23",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "Now subtract the last number",
				MathField.MATH_OPERATION: "23 - 5 = 18",
			},
		],
		ReasoningState.OUTPUT: {
            MathField.ANSWER: "18"
        },
	},
]

# Argument generation demos showing proper use of transition prefixes and internal reasoning
# These demos reflect actual tool usage: claims start with structure prefixes
# and internal_reasoning reflects guidance from subtopic and style choices.
ARGUMENT_DEMOS = [
	{
		ReasoningState.INPUT: {ArgumentField.TOPIC: "renewable energy", ArgumentField.STANCE: ArgumentStance.PRO.value},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should evaluate environmental consequences and sustainability factors. I should present logical arguments based on facts.",
				ArgumentField.CLAIM: "Note that renewable energy significantly reduces carbon emissions compared to fossil fuels",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				ArgumentField.CLAIM: "Therefore solar and wind power have become cost-competitive with traditional energy sources",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze costs, benefits, market effects, and economic implications. I should speak with authority and unwavering confidence.",
				ArgumentField.CLAIM: "Moreover energy independence through renewables improves national security",
			},
		],
		ReasoningState.OUTPUT: {
			ArgumentField.ARGUMENT: "Renewable energy is essential for our future. It provides a sustainable path forward that reduces emissions, costs less, and strengthens our independence."
		},
	},
	{
		ReasoningState.INPUT: {ArgumentField.TOPIC: "remote work", ArgumentField.STANCE: ArgumentStance.PRO.value},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should consider how this affects individuals, groups, and society as a whole. I should build credibility and establish mutual trust.",
				ArgumentField.CLAIM: "First remote work eliminates commuting time and reduces stress",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				ArgumentField.CLAIM: "On the other hand companies save significantly on office space and overhead costs",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should present logical arguments based on facts. I should emphasize information sharing and provide clear insights.",
				ArgumentField.CLAIM: "For example employees report higher productivity when working from home",
			},
		],
		ReasoningState.OUTPUT: {
			ArgumentField.ARGUMENT: "Remote work benefits both employees and employers. Workers gain time and reduce stress, while companies cut costs and often see productivity gains."
		},
	},
	{
		ReasoningState.INPUT: {ArgumentField.TOPIC: "universal basic income", ArgumentField.STANCE: ArgumentStance.PRO.value},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze costs, benefits, market effects, and economic implications. I should present logical arguments based on facts.",
				ArgumentField.CLAIM: "Specifically UBI could reduce poverty by providing a guaranteed income floor for all citizens",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should consider how this affects individuals, groups, and society as a whole. I should build credibility and establish mutual trust.",
				ArgumentField.CLAIM: "Similarly UBI would improve economic security for vulnerable populations",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze costs, benefits, market effects, and economic implications. I should speak with authority and unwavering confidence.",
				ArgumentField.CLAIM: "Essentially UBI represents a fundamental shift toward economic justice",
			},
		],
		ReasoningState.OUTPUT: {
			ArgumentField.ARGUMENT: "Universal Basic Income offers a transformative approach to economic security. It reduces poverty, improves security for vulnerable groups, and represents a fundamental shift toward economic justice."
		},
	},
	{
		ReasoningState.INPUT: {ArgumentField.TOPIC: "AI regulation", ArgumentField.STANCE: ArgumentStance.PRO.value},
		ReasoningState.REASONING: [
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze legal requirements, regulations, and compliance considerations. I should express strong disagreement.",
				ArgumentField.CLAIM: "If AI systems perpetuate discrimination through biased algorithms, regulatory oversight is needed to ensure fairness",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should examine technical requirements, limitations, and implementation challenges. I should speak with authority and unwavering confidence.",
				ArgumentField.CLAIM: "For the sake of preventing harm, government must establish clear safety standards for AI development",
			},
			{
				ControllerOutputParameters.INTERNAL_REASONING: "I should analyze legal requirements, regulations, and compliance considerations. I should present logical arguments based on facts.",
				ArgumentField.CLAIM: "However existing frameworks are insufficient to address rapidly evolving AI capabilities",
			},
		],
		ReasoningState.OUTPUT: {
			ArgumentField.ARGUMENT: "AI regulation is urgently needed. If AI systems perpetuate discrimination, oversight is required. For the sake of preventing harm, government must establish safety standards. However existing frameworks are insufficient for rapidly evolving AI capabilities."
		},
	},
]
