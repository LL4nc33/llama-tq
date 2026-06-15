<script lang="ts">
	import { Brain, Check } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import * as DropdownMenu from '$lib/components/ui/dropdown-menu';
	import * as Tooltip from '$lib/components/ui/tooltip';
	import { REASONING_EFFORT_LEVELS, TOOLTIP_DELAY_DURATION } from '$lib/constants';
	import { SETTINGS_KEYS } from '$lib/constants';
	import { ReasoningEffort } from '$lib/enums';
	import { config, settingsStore } from '$lib/stores/settings.svelte';

	interface Props {
		disabled?: boolean;
	}

	let { disabled = false }: Props = $props();

	let open = $state(false);
	let currentConfig = $derived(config());

	let currentEffort = $derived(
		(currentConfig.reasoningEffort as ReasoningEffort) ?? ReasoningEffort.OFF
	);

	let currentLabel = $derived(
		REASONING_EFFORT_LEVELS.find((level) => level.value === currentEffort)?.label ??
			REASONING_EFFORT_LEVELS[0].label
	);

	// Show a filled icon whenever effort is anything other than 'off'.
	let isActive = $derived(currentEffort !== ReasoningEffort.OFF);

	function selectEffort(value: ReasoningEffort) {
		settingsStore.updateConfig(SETTINGS_KEYS.REASONING_EFFORT, value);
		open = false;
	}
</script>

<DropdownMenu.Root bind:open>
	<DropdownMenu.Trigger name="Reasoning effort" {disabled}>
		<Tooltip.Root delayDuration={TOOLTIP_DELAY_DURATION}>
			<Tooltip.Trigger class="w-full">
				<Button
					variant="ghost"
					size="sm"
					{disabled}
					class="group h-8 gap-1.5 rounded-full px-2.5 {isActive
						? 'text-foreground'
						: 'text-muted-foreground'}"
				>
					<Brain class="h-4 w-4 {isActive ? 'fill-current/15' : ''}" />
					<span class="text-xs font-medium">{currentLabel}</span>
					<span class="sr-only">Reasoning effort: {currentLabel}</span>
				</Button>
			</Tooltip.Trigger>
			<Tooltip.Content>Reasoning effort</Tooltip.Content>
		</Tooltip.Root>
	</DropdownMenu.Trigger>

	<DropdownMenu.Content align="end" class="min-w-40">
		<DropdownMenu.Label>Reasoning effort</DropdownMenu.Label>
		<DropdownMenu.Separator />
		{#each REASONING_EFFORT_LEVELS as level (level.value)}
			<DropdownMenu.Item onclick={() => selectEffort(level.value)}>
				<span class="flex-1">{level.label}</span>
				{#if level.value === currentEffort}
					<Check class="h-4 w-4" />
				{/if}
			</DropdownMenu.Item>
		{/each}
	</DropdownMenu.Content>
</DropdownMenu.Root>
