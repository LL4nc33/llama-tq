# llama-tq — Bare-Metal Setup (fresh Ubuntu 24.04 + NVIDIA GPU)

Real, tested setup path for bringing up llama-tq on a **fresh** Ubuntu 24.04 box
(here: a KVM VM with GPU passthrough, 1× RTX 2060 12 GB, 15 GB RAM, 4 cores).
Every step below is something a clean machine actually needs — discovered while
standing up a brand-new server from zero.

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

The GitHub release ships CUDA-linked `.so`s but **not** the CUDA runtime libs.
On a machine with no CUDA toolkit you still need `libcudart.so.12`,
`libcublas.so.12`, `libnccl.so.2`. Two ways to get them without installing the
full toolkit:

```bash
# download the binary
mkdir -p ~/llama-tq-bin && cd ~/llama-tq-bin
curl -sL https://github.com/LL4nc33/llama-tq/releases/download/<tag>/llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz -o llama-cuda.tar.gz
tar xzf llama-cuda.tar.gz

# gather the runtime libs into one dir
mkdir -p ~/cuda-libs
# cudart + cublas often already present (e.g. from an Ollama install):
cp ~/.local/lib/ollama/cuda_v12/libcudart.so.12*  ~/cuda-libs/ 2>/dev/null
cp ~/.local/lib/ollama/cuda_v12/libcublas*.so.12* ~/cuda-libs/ 2>/dev/null
# nccl from the pip wheel (single-GPU never calls it, but it's a link-time dep):
pip download --no-deps nvidia-nccl-cu12 -d /tmp/nccl
python3 - <<'PY'
import zipfile, glob, os, shutil
w = glob.glob('/tmp/nccl/*.whl')[0]
with zipfile.ZipFile(w) as z:
    for n in z.namelist():
        if 'libnccl.so' in n:
            with z.open(n) as s, open(os.path.expanduser('~/cuda-libs/')+os.path.basename(n),'wb') as d:
                shutil.copyfileobj(s, d)
PY

# run it
BIN=~/llama-tq-bin/llama-<tag>
LD_LIBRARY_PATH=$BIN:~/cuda-libs $BIN/llama-server --version    # should print "found N CUDA devices"
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
already runs):

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
git clone <repo-url> llama-tq && cd llama-tq
git checkout <branch-with-tricks>      # must have SWA-KV flags + gemma4uv vision
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75   # 75 = Turing (RTX 2060)
```

The gemma4uv/gemma4ua vision+audio projector is committed on this branch
(`tools/mtmd/models/gemma4uv.cpp` / `gemma4ua.cpp`) — a plain checkout builds
it in. No manual `git apply` of an mtmd patch is needed; if those files are
missing, you're on the wrong branch.

**Low-RAM warning:** the TurboQuant `fattn` templates are RAM-hungry. On a
15 GB / 4-core VM, build with **`-j1`** (higher `-j` triggers OOM-kills) and
enable ccache. Expect a long compile (the vtq/ktq kernels alone take a while).

```bash
export CCACHE_DIR=~/.ccache
setsid bash -c 'cmake --build build --target llama-server -j1 > /tmp/build.log 2>&1' < /dev/null &
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
  --host 0.0.0.0 --port 8791 --jinja --flash-attn on \
  -c 131072 -ngl 99 --parallel 1 -ub 512 \
  -ctk q8_0 -ctv q8_0 \
  --no-context-shift --reasoning off
```

**gemma-4 gotchas (learned the hard way):**
- KV `ktq2_1`/`vtq2_3` produce `<unused49>` garbage on gemma-4 (SWA layers).
  Use `q8_0/q8_0`, or — with a build that has them — `-ctk ktq2_1 -ctv vtq2_3`
  **plus** `--cache-type-k-swa f16 --cache-type-v-swa vtq3` to keep SWA layers safe.
- Vision needs the `gemma4uv` mmproj. The matching `gemma4uv`/`gemma4ua`
  projector support is already committed on the tricks branch (§3.2), so a
  source build has it; only an older *prebuilt release* may lack it. Image
  detail depends on `--image-max-tokens` (budgets: 70/140/280/560/1120; use
  **1120** for OCR / small text) with `-ub 2048 -b 2048`.
- A mismatched MTP draft model logs `failed to decode draft batch, ret = -1`
  every token and craters TG — drop `--model-draft` if the draft doesn't match.

Test:
```bash
curl -s localhost:8791/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hauptstadt Österreich? Ein Wort."}],"max_tokens":10}'
```

**ngram-spec note:** `--spec-type ngram-cache` is lossless and content-dependent —
the cache learns from generated text, so short prompts show no speed-up (draft
stats read `#gen drafts = 0`). On repetitive content (code, lists, repeated
structure) it kicks in: measured ~33 t/s → ~59 t/s (≈1.8×) on gemma-4-12B / RTX 2060.

---

## 6. Auto-start on boot (systemd)

Put the deploy command in a script (`~/deploy-gemma4-12b.sh`, `chmod +x`) and
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
ExecStart=/home/<user>/deploy-gemma4-12b.sh
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
