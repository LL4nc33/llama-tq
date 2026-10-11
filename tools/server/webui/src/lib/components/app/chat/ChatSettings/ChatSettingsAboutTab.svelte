<script lang="ts">
	import { ExternalLink } from '@lucide/svelte';
	import { serverProps } from '$lib/stores/server.svelte';

	// release version from the release workflow (empty for builds from source); build_info is "b<build>-<commit>"
	let release = $derived(serverProps()?.version ?? '');
	let build = $derived.by(() => {
		const info = serverProps()?.build_info ?? '';
		const m = info.match(/^b(\d+)-(\w+)$/);
		return m ? `build ${m[1]}, commit ${m[2]}` : info;
	});

	const links = [
		{
			label: 'llama-tq',
			href: 'https://github.com/LL4nc33/llama-tq',
			note: 'this fork: TurboQuant KV cache, fine-tuning on quantized models, multi-GPU without P2P'
		},
		{
			label: 'llama.cpp',
			href: 'https://github.com/ggml-org/llama.cpp',
			note: 'upstream project this server and web interface are based on'
		}
	];
</script>

<div class="space-y-6 text-sm">
	<div class="space-y-1">
		<p class="text-base font-medium">llama-tq</p>
		<p class="text-muted-foreground">
			A fork of llama.cpp for consumer NVIDIA GPUs. The server and this web interface come from
			upstream llama.cpp.
		</p>
		{#if serverProps()}
			<p class="text-muted-foreground">
				Version: <code>{release || 'development build'}</code>{#if build}
					({build}){/if}
			</p>
		{/if}
	</div>

	<div class="space-y-3">
		<p class="font-medium">Source code</p>
		{#each links as link (link.href)}
			<div>
				<a
					href={link.href}
					target="_blank"
					rel="noopener noreferrer"
					class="inline-flex items-center gap-1 font-medium underline-offset-4 hover:underline"
				>
					{link.label}
					<ExternalLink class="h-3.5 w-3.5" />
				</a>
				<p class="text-muted-foreground">{link.note}</p>
			</div>
		{/each}
	</div>

	<div class="space-y-2">
		<p class="font-medium">License</p>
		<p class="text-muted-foreground">
			MIT License, Copyright (c) 2023-2026 The ggml authors. llama-tq is distributed under the same
			license.
		</p>
		<a
			href="https://github.com/LL4nc33/llama-tq/blob/turboquant/LICENSE"
			target="_blank"
			rel="noopener noreferrer"
			class="inline-flex items-center gap-1 underline-offset-4 hover:underline"
		>
			Full license text
			<ExternalLink class="h-3.5 w-3.5" />
		</a>
	</div>
</div>
