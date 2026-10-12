# llama-tq — Bare-Metal Setup (fresh Ubuntu 24.04 + NVIDIA GPU)

Setup path for bringing up llama-tq on a **fresh** Ubuntu 24.04 box, tested in
July 2026 on a KVM VM with GPU passthrough (1× RTX 2060 12 GB, 15 GB RAM, 4 cores).
Every step below is something a clean machine actually needed.

There are **two ways** to get a running server:
- **Fast path (A):** download a prebuilt release binary — no compiler, ~20 min.
- **Full path (B):** build from source — needed for the TurboQuant KV tricks,
  ngram-spec, and gemma-4 vision (`gemma4uv`) that a given release may predate.

---

## 0. Prerequisites check

```bash
lspci | grep -i nvidia          # GPU visible to the OS?
nvidia-smi                      # driver talking to GPU?
nproc; free -h; df -h /         # cores / RAM / disk
```

If `nvidia-smi` fails with *"couldn't communicate with the NVIDIA driver"* but
the driver package is installed, the kernel module is just not loaded — see §1.

---

## 1. NVIDIA driver (module not loaded)

Symptom: `nvidia-driver-XXX` is installed (`dpkg -l | grep nvidia-driver`) and
`dkms status` shows it built for the running kernel, but `lsmod | grep nvidia`
is empty.

```bash
sudo modprobe nvidia
sudo modprobe nvidia_uvm
nvidia-smi          # should now list the GPU

# make it load on every boot:
echo -e "nvidia\nnvidia_uvm\nnvidia_modeset" | sudo tee /etc/modules-load.d/nvidia.conf
```

---

## 2. Fast path (A): prebuilt release binary

Download the CUDA archive (from the next release on, one per GPU generation) from the
[releases page](https://github.com/LL4nc33/llama-tq/releases); what it needs (CUDA runtime libraries, driver) is
listed in [build.md, Release archives](../build.md#release-archives). The current release v2026.10.06 still needs
`libnccl.so.2` and `LD_LIBRARY_PATH` pointing at its own directory; the next release needs neither.

```bash
mkdir -p ~/llama-tq-bin && cd ~/llama-tq-bin
tar xzf ~/Downloads/<archive>.tar.gz
BIN=~/llama-tq-bin/llama-<tag>

# v2026.10.06 only: libnccl.so.2 from the pip wheel (single-GPU never calls it, but it is a link-time dependency)
pip download --no-deps nvidia-nccl-cu12 -d nccl-wheel
python3 -m zipfile -e nccl-wheel/*.whl nccl-wheel/x
cp $(find nccl-wheel/x -name 'libnccl.so*') $BIN/
LD_LIBRARY_PATH=$BIN $BIN/llama-server --version    # should print "found N CUDA devices"
```

**Limitation:** an older release may lack `--cache-type-k-swa`/`-v-swa`,
gemma-4 ngram-spec, and the `gemma4uv` vision projector. If the server logs
`speculative decoding not supported by this context` or the mmproj fails to
load, you need the full build (path B).

---

## 3. Full path (B): build from source

### 3.1 CUDA toolkit (match the runtime, not apt's default)

apt's `nvidia-cuda-toolkit` is 12.0 on 24.04 — too old to match a 12.8 runtime
/ recent driver. Use NVIDIA's repo and install **only the toolkit** (the driver
already runs; for Blackwell use 12.9 or newer, see [build.md](../build.md#build-llama-tq-locally)):

```bash
curl -sL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o /tmp/cuda-keyring.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-12-8 cmake ccache build-essential
export PATH=/usr/local/cuda-12.8/bin:$PATH
nvcc --version
```

### 3.2 clone + build

```bash
git clone https://github.com/LL4nc33/llama-tq && cd llama-tq
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75   # 75 = Turing (RTX 2060)
```

The gemma4uv/gemma4ua vision+audio projector is in the source tree
(`tools/mtmd/models/gemma4uv.cpp` / `gemma4ua.cpp`), so a plain clone builds
it in. No manual `git apply` of an mtmd patch is needed; if those files are
missing, your checkout is older than the current source.

**Low-RAM warning:** the TurboQuant `fattn` templates are RAM-hungry. On 16 GB
RAM build with **`-j2`** (each nvcc job for the TurboQuant templates takes about
3-4 GB) and enable ccache. Expect a long compile.

```bash
cmake --build build --target llama-server -j2
```

---

## 4. Models (HuggingFace)

Anonymous HF downloads are throttled (~3 MB/s regardless of your line speed);
an HF token lifts that. File names change over time — list the repo first:

```bash
curl -sL "https://huggingface.co/api/models/unsloth/gemma-4-12b-it-GGUF/tree/main" \
  | python3 -c "import sys,json;[print(f['path']) for f in json.load(sys.stdin) if f['path'].endswith('.gguf')]"

mkdir -p ~/models && cd ~/models
curl -fsSL https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/gemma-4-12b-it-UD-Q4_K_XL.gguf -o model.gguf
curl -fsSL https://huggingface.co/unsloth/gemma-4-12b-it-GGUF/resolve/main/mmproj-F16.gguf -o mmproj.gguf
```

---

## 5. Deploy (single 12 GB GPU, verified-coherent config)

```bash
export CUDA_VISIBLE_DEVICES=0
llama-server \
  -m ~/models/model.gguf \
  --host 0.0.0.0 --port 8080 --jinja --flash-attn on \
  -c 131072 -ngl 99 --parallel 1 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  --no-context-shift --reasoning off
```

**gemma-4 gotchas (learned the hard way):**
- KV cache: `q8_0/q8_0`, or `q8_0/vtq4_1` for more context; accuracy per type and model in
  [turboquant.md](../turboquant.md#accuracy). Older notes here recommended `ktq2_1`/`vtq2_3` with SWA overrides
  against `<unused49>` output; `vtq2_3` is a trellis type that decodes at 1-3 t/s.
- Vision needs the `gemma4uv` mmproj. The matching `gemma4uv`/`gemma4ua`
  projector support is in the source tree (§3.2), so a
  source build has it; only an older *prebuilt release* may lack it. Image
  detail depends on `--image-max-tokens` (budgets: 70/140/280/560/1120; use
  **1120** for OCR / small text) with `-ub 2048 -b 2048`.
- A mismatched MTP draft model logs `failed to decode draft batch, ret = -1`
  every token and craters TG — drop `--model-draft` if the draft doesn't match.

Test:
```bash
curl -s localhost:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Capital of Austria? One word."}],"max_tokens":10}'
```

**ngram-spec note:** `--spec-type ngram-cache` is lossless and content-dependent —
the cache learns from generated text, so short prompts show no speed-up (draft
stats read `#gen drafts = 0`). On repetitive content (code, lists, repeated
structure) it kicks in: measured ~33 t/s → ~59 t/s (≈1.8×) on gemma-4-12B (2026-07, 1× RTX 2060 12 GB).

---

## 6. Auto-start on boot (systemd)

Put the deploy command in a script (`/usr/local/bin/deploy-gemma4-12b.sh`, `chmod +x`) and
wrap it in a system service so it survives reboots and restarts on crash:

```ini
# /etc/systemd/system/gemma4-server.service
[Unit]
Description=llama-tq gemma-4-12B server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<user>
ExecStart=/usr/local/bin/deploy-gemma4-12b.sh
Restart=on-failure
RestartSec=10
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable gemma4-server      # boot-persistent
sudo systemctl start gemma4-server
sudo journalctl -u gemma4-server -f      # follow startup / model load
```

Model load takes ~40–60 s; `systemctl is-active` reads `active` immediately, but
`/health` only returns `ok` once the weights + mmproj are loaded.
