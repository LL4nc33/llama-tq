# llama-tq/examples/training

`llama-finetune` (source: [finetune.cpp](finetune.cpp)) trains LoRA adapters directly on quantized GGUF models, or the
F32 tensors of a model without LoRA, on CPU and CUDA. Options, data format, supported layers and results:
[docs/finetune.md](../../docs/finetune.md). A complete, reproducible task with data generator, training commands,
evaluation script and a PyTorch reference: [termine/](termine/README.md).

```sh
./build/bin/llama-finetune -m model.gguf -f train.jsonl -o task.gguf -ngl 99 -c 512 \
  --lora-train-target '^blk\.[0-9]+\.attn_(q|k|v|output)\.weight$' --lora-train-rank 16
# writes task.lora.gguf; serve it with: llama-server -m model.gguf --lora task.lora.gguf
```
