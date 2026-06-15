import { ReasoningEffort } from '$lib/enums';

/**
 * Reasoning effort levels shown in the chat-form selector.
 * 'off' omits the chat_template_kwarg entirely (model default applies).
 */
export interface ReasoningEffortLevel {
	value: ReasoningEffort;
	label: string;
}

export const REASONING_EFFORT_LEVELS: ReasoningEffortLevel[] = [
	{ value: ReasoningEffort.OFF, label: 'Off' },
	{ value: ReasoningEffort.LOW, label: 'Low' },
	{ value: ReasoningEffort.MEDIUM, label: 'Medium' },
	{ value: ReasoningEffort.HIGH, label: 'High' },
	{ value: ReasoningEffort.MAX, label: 'Max' }
];

/**
 * Per-level reasoning token budget sent as `thinking_budget_tokens`.
 * -1 = unlimited. 'off' has no entry (no budget sent). Values match upstream.
 */
export const REASONING_EFFORT_TOKENS: Record<string, number> = {
	[ReasoningEffort.LOW]: 512,
	[ReasoningEffort.MEDIUM]: 2048,
	[ReasoningEffort.HIGH]: 8192,
	[ReasoningEffort.MAX]: -1
};
