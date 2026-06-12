<script lang="ts">
	// Live denoise preview for text-diffusion models (DiffusionGemma).
	// Renders the current canvas (REPLACE per step) with a confidence heatmap:
	// tokens are tinted by how early they settled (cool = early/confident,
	// warm = late, dim = still noisy this step).
	interface Props {
		preview: {
			canvas: string;
			step: number;
			total: number;
			settled?: number[];
		};
	}

	let { preview }: Props = $props();

	const pct = $derived(
		preview.total > 0 ? Math.round((preview.step / preview.total) * 100) : 0
	);

	// Map a settle-step to a hue (cool early → warm late). 0 = not settled yet.
	function tint(settleStep: number, total: number): string {
		if (!settleStep || settleStep <= 0) return 'opacity:0.35;'; // still noisy
		const frac = total > 0 ? Math.min(settleStep / total, 1) : 0;
		const hue = Math.round(210 - 210 * frac); // 210 (blue) → 0 (red)
		return `color:hsl(${hue},70%,55%);`;
	}

	// Split the canvas into whitespace-preserving chunks so we can tint roughly
	// per word. The settled[] array is per-token; we don't have token offsets in
	// the UI, so we approximate by distributing settle info across visible words.
	const words = $derived(preview.canvas.split(/(\s+)/));
	const settled = $derived(preview.settled ?? []);
	const total = $derived(preview.total);

	function wordTint(i: number): string {
		if (!settled.length) return '';
		// approximate token index from word index
		const idx = Math.min(Math.floor((i / Math.max(words.length, 1)) * settled.length), settled.length - 1);
		return tint(settled[idx] ?? 0, total);
	}
</script>

<div class="diffusion-canvas">
	<div class="diffusion-canvas__bar">
		<span class="diffusion-canvas__label">✨ denoising</span>
		<span class="diffusion-canvas__step">step {preview.step}/{preview.total}</span>
		<div class="diffusion-canvas__progress">
			<div class="diffusion-canvas__progress-fill" style="width:{pct}%"></div>
		</div>
	</div>
	<div class="diffusion-canvas__text">
		{#each words as word, i (i)}<span style={wordTint(i)}>{word}</span>{/each}
	</div>
</div>

<style>
	.diffusion-canvas {
		border: 1px solid var(--border, rgba(128, 128, 128, 0.25));
		border-radius: 0.5rem;
		padding: 0.6rem 0.75rem;
		margin: 0.25rem 0;
		background: var(--surface-1, rgba(128, 128, 128, 0.05));
	}
	.diffusion-canvas__bar {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		margin-bottom: 0.45rem;
		font-size: 0.78rem;
		opacity: 0.85;
	}
	.diffusion-canvas__label {
		font-weight: 600;
	}
	.diffusion-canvas__step {
		font-variant-numeric: tabular-nums;
		opacity: 0.7;
	}
	.diffusion-canvas__progress {
		flex: 1;
		height: 4px;
		border-radius: 2px;
		background: rgba(128, 128, 128, 0.2);
		overflow: hidden;
	}
	.diffusion-canvas__progress-fill {
		height: 100%;
		background: linear-gradient(90deg, hsl(210, 70%, 55%), hsl(280, 70%, 55%));
		transition: width 0.15s ease;
	}
	.diffusion-canvas__text {
		font-family: var(--font-mono, ui-monospace, monospace);
		font-size: 0.82rem;
		line-height: 1.5;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 18rem;
		overflow-y: auto;
	}
</style>
