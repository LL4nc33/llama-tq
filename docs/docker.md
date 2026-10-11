# Docker

## Published images

This fork publishes CPU images for `linux/amd64`:

| Image | Contents |
| --- | --- |
| `ghcr.io/ll4nc33/llama-tq:server` | `llama-server` |
| `ghcr.io/ll4nc33/llama-tq:light` | `llama-cli` and `llama-completion` |
| `ghcr.io/ll4nc33/llama-tq:full` | all tools plus the Python scripts to convert and quantize models |

```bash
docker run -v /path/to/models:/models -p 8080:8080 ghcr.io/ll4nc33/llama-tq:server \
  -m /models/model.gguf --host 0.0.0.0 --port 8080
```

The TurboQuant KV types need CUDA, so for a GPU use a CUDA image built locally (below) or a
[release archive](https://github.com/LL4nc33/llama-tq/releases) for your GPU generation. The images of upstream
llama.cpp (`ghcr.io/ggml-org/llama.cpp`) do not contain the changes of this fork.

## CUDA image, built locally

Needs the [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit) on the host.

```bash
# 75 Turing, 86 Ampere, 89 Ada, 120 Blackwell; several as "86;89"
docker build -t llama-tq:server-cuda --target server --build-arg CUDA_DOCKER_ARCH=86 -f .devops/cuda.Dockerfile .

docker run --gpus all -v /path/to/models:/models -p 8080:8080 llama-tq:server-cuda \
  -m /models/model.gguf -ngl 99 -fa on -ctk q8_0 -ctv vtq4_1 --host 0.0.0.0 --port 8080
```

Build arguments: `CUDA_VERSION` (default `12.8.1`), `CUDA_DOCKER_ARCH` (default: all architectures of the CMake build,
which takes much longer). The targets `full`, `light` and `server` match the published images.

Other backends (ROCm, MUSA, SYCL, Vulkan, OpenVINO) have Dockerfiles in [.devops/](../.devops/) as in upstream; they
are built and used the same way and contain no TurboQuant KV types.
