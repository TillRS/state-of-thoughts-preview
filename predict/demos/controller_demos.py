"""
Controller Demo Constants

This module contains demonstration examples for the TreeOfThoughtsController
that show when to use FINISH vs CONTINUE_REASONING tools based on the completeness
of the reasoning process.

These demos are structured to match the controller's signature format and guide
the model to make appropriate decisions about reasoning completion.
"""

# Standard library imports
from typing import Any

# Local imports
from predict import (
	ControllerActionParameters,
	ControllerConfig,
	ControllerContinueReasoningChoice,
)
from signatures import ArgumentStance
from signatures.example_signatures import ArgumentField, MathField
from tree.tree_constants import ReasoningState

# Demos showing when to ControllerContinueReasoningChoice.FINISH reasoning (reasoning is sufficient)
CONTROLLER_FINISH_DEMOS: list[dict[str, Any]] = [
	{
		# Complete arithmetic solution
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is 15 * 8?",
			MathField.MATH_OPERATION: ["I need to multiply 15 by 8", "15 * 8 = 120"],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The multiplication has been completed correctly and the answer is clear. No further reasoning is needed.",
			ControllerActionParameters.ACTION: "finish",
		},
	},
	{
		# Complete linear equation solution
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Solve 2x + 3 = 11",
			MathField.MATH_OPERATION: [
				"I need to isolate x by subtracting 3 from both sides.",
				"2x + 3 - 3 = 11 - 3",
				"2x = 8",
				"Dividing both sides by 2: x = 4",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The equation has been completely solved. I found x = 4 and verified the steps. The solution is complete.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
		},
	},
	{
		# Complete geometry calculation
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "What is the area of a circle with radius 4?",
			MathField.MATH_OPERATION: [
				"The formula for the area of a circle is A = πr².",
				"Given r = 4, so A = π * 4² = π * 16 = 16π.",
				"The area is 16π square units.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I have correctly applied the area formula and calculated the result. The answer is complete and clearly stated.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
		},
	},
]


# Demos showing when to ControllerContinueReasoningChoice.CONTINUE_REASONING (more reasoning needed)
CONTROLLER_CONTINUE_DEMOS: list[dict[str, Any]] = [
	{
		# Problem identified but not solved
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Solve x^3 - 10x + 5 = 0",
			MathField.MATH_OPERATION: ["This is a cubic equation in standard form."],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 4,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've identified this as a cubic equation, but I haven't chosen a solution method or begun solving it. I need to continue reasoning to find the solution.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Method identified but not applied
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Find the derivative of f(x) = x³ + 2x² - x + 1",
			MathField.MATH_OPERATION: ["I need to find the derivative using the power rule."],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I know I need to use the power rule, but I haven't actually applied it to each term yet. More reasoning steps are needed to compute the derivative.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Partial geometric proof
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Prove that the sum of angles in any triangle is 180°",
			MathField.MATH_OPERATION: [
				"I need to prove this geometric theorem.",
				"Let me consider a triangle ABC with angles α, β, and γ.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 4,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've set up the proof with a triangle ABC, but I haven't provided the actual proof steps. I need to continue with the geometric reasoning.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Incomplete analysis
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Is 91 a prime number?",
			MathField.MATH_OPERATION: [
				"A prime number has exactly two factors: 1 and itself.",
				"I need to check if 91 has any factors other than 1 and 91.",
				"I should test divisibility by primes up to √91 ≈ 9.5.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've outlined the approach for checking primality, but I haven't actually tested the divisibility. I need to continue reasoning to check each potential factor.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Incomplete factoring
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Factor x^2 - 9",
			MathField.MATH_OPERATION: [
				"This looks like a difference of squares pattern.",
				"The difference of squares formula is a^2 - b^2 = (a + b)(a - b).",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've identified this as a difference of squares and recalled the formula, but I haven't applied it to actually factor x² - 9.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Partial FOIL expansion
		ReasoningState.INPUT: {
			MathField.MATH_PROBLEM: "Expand (x + 2)(x - 3)",
			MathField.MATH_OPERATION: [
				"I'll use the FOIL method.",
				"First: x * x = x^2",
				"Outer: x * (-3) = -3x",
				"Inner: 2 * x = 2x",
				"Last: 2 * (-3) = -6",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've computed all the FOIL terms, but I haven't combined them into the final expanded form. I need to add the terms together.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
]


# Argument-specific demos showing diverse subtopic, style, and structure choices
ARGUMENT_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Economic Impact + Knowledge + Level-of-Detail - opening with specifics
		# First step uses "Specifically" to dive into economic details
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "UBI should be implemented nationally",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Specifically UBI could reduce poverty by providing a guaranteed income floor for all citizens.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've made one economic point, but I should analyze more specific economic impacts like costs, funding mechanisms, and market effects to build a compelling argument. I should present logical arguments based on facts.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Economic Impact",
				"style": "Knowledge",
				"structure": "Cause",
			},
		},
	},
	{
		# Social Impact + Trust + Contrast - mid-reasoning example
		# Previous step established a point, now using "Cause" to show consequence, then "Contrast" for next step
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Tech companies should face stricter data protection rules",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"First current privacy laws are insufficient for protecting user data.",
				"Therefore companies currently collect vast amounts of personal information with minimal oversight.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've established the problem and its consequence, but need to explore how stricter regulations would affect people and communities. I should build credibility and establish mutual trust. A contrasting perspective would strengthen the argument.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Social Impact",
				"style": "Trust",
				"structure": "Contrast",
			},
		},
	},
	{
		# Legal & Regulatory + Power + Instantiation - showing completion
		# Previous steps show proper prefixes and internal reasoning
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Self-driving cars should require federal certification",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Generally current state-by-state approaches create regulatory fragmentation.",
				"On the other hand federal oversight would establish uniform safety standards across all jurisdictions.",
				"For example the National Highway Traffic Safety Administration could mandate standardized testing protocols for all autonomous vehicles.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've covered regulatory analysis with concrete examples and built a strong authoritative argument. The reasoning is comprehensive enough to finish.",
			ControllerActionParameters.ACTION: "finish",
			ControllerActionParameters.ARGUMENTS: {},
		},
	},
	{
		# Technical Feasibility + Power + Purpose - diverse combination
		# Should show "For the sake of" prefix from Purpose structure
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Government should mandate solar panels on new buildings",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Earlier renewable energy infrastructure was seen as optional, but climate change demands immediate action.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 4,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've established the temporal context, but need to examine technical requirements and implementation challenges while maintaining an authoritative tone. I should focus on the purpose behind this policy.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Technical Feasibility",
				"style": "Power",
				"structure": "Purpose",
			},
		},
	},
	{
		# Environmental Impact + Support + Similarity - showing empathy
		# Should show "Similarly" prefix from Similarity structure
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Businesses should eliminate single-use plastics",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"At the same time plastic pollution threatens marine ecosystems and human health.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've established the environmental threat, but should evaluate environmental consequences and sustainability factors while being emotionally supportive. I should emphasize our shared concern for the planet.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Environmental Impact",
				"style": "Support",
				"structure": "Similarity",
			},
		},
	},
	{
		# Legal & Regulatory + Conflict + Concession - confrontational approach
		# Should show "However" prefix from Concession structure
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Platforms should be held liable for user-generated content",
			ArgumentField.STANCE: ArgumentStance.ANTI,
			ArgumentField.CLAIM: [
				"Essentially free speech principles are fundamental to democratic discourse.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've restated the core principle, but need to analyze legal requirements and regulations while taking a more confrontational stance. I should acknowledge counterarguments.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Legal & Regulatory",
				"style": "Conflict",
				"structure": "Concession",
			},
		},
	},
	{
		# Economic Impact + Knowledge + Conjunction - adding information
		# Previous step established a point, now adding more information with "Moreover"
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "The government should regulate AI development",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"If AI systems perpetuate discrimination through biased algorithms, regulatory oversight is needed to ensure fairness.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've established a conditional relationship, but should analyze more economic impacts like costs, benefits, and market effects. I should present logical arguments based on facts and add more information to strengthen the argument.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Economic Impact",
				"style": "Knowledge",
				"structure": "Conjunction",
			},
		},
	},
	{
		# Social Impact + Trust + Instantiation - providing examples
		# Should show "For example" prefix from Instantiation (Specific) structure
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Universal healthcare should be implemented",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Rather than maintaining fragmented insurance systems, universal healthcare would improve access to medical services for underserved populations.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've presented an alternative approach, but should consider how this affects people and communities while building credibility. I should provide concrete examples to illustrate the point.",
			ControllerActionParameters.ACTION: "select_subtopic_style_structure",
			ControllerActionParameters.ARGUMENTS: {
				"subtopic": "Social Impact",
				"style": "Trust",
				"structure": "Instantiation (Specific)",
			},
		},
	},
]

# Style-only demos (for generative controllers with style action space + finish)
# Uses intervene_on_next_reasoning_step action with style parameter
STYLE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Statistical context should encourage style variation
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Climate change mitigation",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Research shows that renewable energy costs have dropped 89% since 2010.",
				"Studies indicate solar power is now cheaper than fossil fuels in most markets.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've presented statistical evidence, but my argument could benefit from a more engaging, figurative approach to connect emotionally with readers.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"style": "Figurative Language",
			},
		},
	},
	{
		# Figurative language should encourage statistical balance
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Universal healthcare",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Healthcare is like oxygen - everyone needs it to survive and thrive.",
				"A healthy population is the foundation upon which all other progress is built.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've used figurative language effectively, but now I need concrete data and statistics to strengthen my argument with evidence.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"style": "Statistical & Data-Driven",
			},
		},
	},
	{
		# Complete argument should finish
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Education funding",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Education is an investment in our collective future.",
				"Studies show that every dollar invested in education returns $7-10 to the economy.",
				"Therefore, increased education funding is both morally right and economically sound.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The argument is complete with both figurative appeal and statistical evidence, concluding with a clear thesis. No further reasoning is needed.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
			ControllerActionParameters.ARGUMENTS: {},
		},
	},
]


# Structure-only demos (for generative controllers with structure action space + finish)
# Uses intervene_on_next_reasoning_step action with structure parameter
STRUCTURE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Claim needs causal evidence
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Electric vehicle adoption",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Electric vehicles are the future of transportation.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've made a claim but haven't explained why EVs are the future. I need to add causal reasoning to explain the underlying factors.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"structure": "Causal Reasoning",
			},
		},
	},
	{
		# After claim and cause, add contrast
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Remote work policies",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Remote work increases employee productivity.",
				"Because workers save commute time and have more control over their environment, they are more focused and efficient.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've established a claim with causal reasoning, but I should acknowledge the contrasting perspective to strengthen my argument.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"structure": "Contrast",
			},
		},
	},
	{
		# Complete structural argument should finish
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Public transit investment",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Cities should prioritize public transit expansion.",
				"Because public transit reduces traffic congestion and emissions, it improves quality of life for all residents.",
				"On the other hand, while cars offer flexibility, their societal costs outweigh individual benefits.",
				"Therefore, investing in public transit is the responsible choice for sustainable urban development.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The argument has a clear claim, causal reasoning, contrast, and conclusion. The structure is complete and ready to finish.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
			ControllerActionParameters.ARGUMENTS: {},
		},
	},
]


# Style + Structure combined demos (for generative controllers with both action spaces)
# Uses intervene_on_next_reasoning_step action with both style and structure parameters
STYLE_STRUCTURE_CONTROLLER_DEMOS: list[dict[str, Any]] = [
	{
		# Early stage - needs both style and structure
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Renewable energy investment",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"We should increase investment in renewable energy.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I have only a basic claim. I need to add substance with evidence. Using figurative language with causal reasoning will make the argument both engaging and logical.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"style": "Figurative Language",
				"structure": "Causal Reasoning",
			},
		},
	},
	{
		# After figurative+causal, balance with statistical+contrast
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "AI regulation",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"AI systems must be regulated to prevent harm.",
				"Like a powerful river, AI needs proper channels to direct its force productively rather than destructively.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've used figurative language. Now I should balance with data and address counterarguments through statistical evidence and contrasting views.",
			ControllerActionParameters.ACTION: "intervene_on_next_reasoning_step",
			ControllerActionParameters.ARGUMENTS: {
				"style": "Statistical & Data-Driven",
				"structure": "Contrast",
			},
		},
	},
	{
		# Complete argument ready to finish
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Carbon tax implementation",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"A carbon tax is essential for addressing climate change.",
				"Just as we price harmful substances like tobacco to reduce consumption, pricing carbon emissions creates natural market incentives.",
				"Studies show that countries with carbon taxes have reduced emissions by 15-20% compared to those without.",
				"While critics argue costs hurt businesses, evidence from British Columbia shows economic growth remained strong after carbon tax implementation.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The argument includes figurative language, statistical evidence, causal reasoning, and contrast. It is comprehensive and complete.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
			ControllerActionParameters.ARGUMENTS: {},
		},
	},
]


# Combined demo set for controller (continue/finish only - math demos)
CONTROLLER_DEMOS: list[dict[str, Any]] = (
	CONTROLLER_FINISH_DEMOS + CONTROLLER_CONTINUE_DEMOS
)


# Argument-based continue/finish only demos (for controllers with no action spaces)
# Uses ArgumentField but only continue_reasoning and finish actions
ARGUMENT_CONTINUE_FINISH_DEMOS: list[dict[str, Any]] = [
	{
		# Demo #1: Incomplete argument - needs more reasoning
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Universal basic income",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"UBI could help reduce poverty rates.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 3,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've only made one claim without supporting evidence or explanation. I need to continue reasoning to develop this argument further.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Demo #2: Still needs more content
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Electric vehicle subsidies",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Government subsidies for EVs can accelerate the transition to clean transportation.",
				"These subsidies reduce the upfront cost barrier that prevents many consumers from choosing EVs.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 2,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "I've provided some reasoning, but the argument lacks a conclusion and could benefit from addressing counterarguments. More reasoning steps are available.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.CONTINUE_REASONING,
		},
	},
	{
		# Demo #3: Complete argument - ready to finish
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Renewable energy investment",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Investing in renewable energy is essential for long-term economic stability.",
				"Fossil fuel prices are volatile and subject to geopolitical instability, while renewable sources provide predictable costs.",
				"Countries that have invested heavily in renewables, like Denmark and Germany, have created millions of green jobs.",
				"Therefore, transitioning to renewable energy is both economically prudent and environmentally necessary.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "The argument presents a clear claim, supporting evidence with examples, and a conclusive statement. It is complete and ready to finish.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
		},
	},
	{
		# Demo #4: Another complete argument
		ReasoningState.INPUT: {
			ArgumentField.TOPIC: "Public education funding",
			ArgumentField.STANCE: ArgumentStance.PRO,
			ArgumentField.CLAIM: [
				"Increased public education funding leads to better societal outcomes.",
				"Research shows that well-funded schools have higher graduation rates and better test scores.",
				"Education reduces crime rates and increases earning potential, benefiting the entire economy.",
				"In conclusion, investing in public education is an investment in our collective future.",
			],
			ControllerConfig.NUMBER_OF_ADDITIONAL_REASONING_STEPS: 1,
		},
		ReasoningState.OUTPUT: {
			ControllerActionParameters.CONSIDERATIONS: "This argument has a thesis, evidence, analysis, and conclusion. The reasoning is complete and no further steps are needed.",
			ControllerActionParameters.ACTION: ControllerContinueReasoningChoice.FINISH,
		},
	},
]
