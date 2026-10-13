"""DeepSeek-V4.1-Flash on four Modal B300s with vLLM.

    modal secret create huggingface HF_TOKEN=hf_...   # once
    modal run deepseek_v41_flash.py --download-only  # cache weights on CPU
    modal deploy deepseek_v41_flash.py           # publish the endpoint
    modal run deepseek_v41_flash.py              # send one test request
    modal app stop deepseek-v41-flash-starter     # stop compute; volumes remain

The server is private: Modal's edge checks a proxy token before a request reaches
the GPU. Create one with `modal workspace proxy-tokens create` and send it as
`Authorization: Bearer wk-....ws-...`.
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

import modal

MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "5254448103b6254fc646dacbca1cbbf20e9216c1"  # checked 2026-10-12, about 511 GB
SERVED_NAME = "deepseek-v4.1-flash"
# vLLM nightly fe4df3df (2026-10-12), pinned to the Linux amd64 image.
VLLM_IMAGE = "vllm/vllm-openai@sha256:02b51ccd46ab3fe7e0abb42aa55e3f27a52daae40d9b5c4dbd3ebb86902a2d7c"
GPU = "B300:4"
PORT = 8000
MINUTES = 60

# Model-specific settings from the vLLM recipe; tune against your workload.
VLLM_ARGS = [
    "--revision", REVISION,
    "--served-model-name", SERVED_NAME,
    "--trust-remote-code",
    "--tokenizer-mode", "deepseek_v41",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--tensor-parallel-size", "4",
    "--language-model-only",
    "--engram-config", '{"cpu_offload":false}',
    "--kv-cache-dtype", "fp8",
    "--gpu-memory-utilization", "0.92",
    "--max-model-len", "262144",
    "--max-num-seqs", "32",
    "--max-cudagraph-capture-size", "192",
    "--max-num-batched-tokens", "8192",
    "--attention-config", '{"backend":"FLASHINFER_MLA_SPARSE_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}',
    "--speculative-config", '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}',
    "--reasoning-parser", "deepseek_v41",
    "--tool-call-parser", "deepseek_v41", "--enable-auto-tool-choice",
]

# The vLLM image ships its own Python for vLLM but no plain `python` on PATH, so Modal
# gets a separate interpreter for its own runtime (add_python).
image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .env({
        "HF_XET_HIGH_PERFORMANCE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "7200",
        "FLASHINFER_WORKSPACE_BASE": "/root/.cache/vllm",
        "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR": "/root/.cache/vllm/flashinfer-autotune",
        "TRITON_CACHE_DIR": "/root/.cache/vllm/triton",
        "CUDA_CACHE_PATH": "/root/.cache/vllm/cuda",
    })
)

app = modal.App("deepseek-v41-flash-starter")
weights = modal.Volume.from_name("deepseek-v41-flash-hf", create_if_missing=True)


@app.function(image=image, cpu=8, memory=32768, timeout=7200,
              volumes={"/root/.cache/huggingface": weights},
              secrets=[modal.Secret.from_name("huggingface")])
def download():
    """Cache the public checkpoint on a CPU container before renting GPUs."""
    subprocess.run(["hf", "download", MODEL, "--revision", REVISION], check=True)
    weights.commit()


@app.server(
    image=image,
    gpu=GPU,
    cpu=16,
    memory=256 * 1024,  # MiB; host RAM for loading and worker processes
    port=PORT,
    volumes={
        "/root/.cache/huggingface": weights,
        "/root/.cache/vllm": modal.Volume.from_name("deepseek-v41-flash-compile", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface")],
    startup_timeout=120 * MINUTES,  # allow kernel JIT and CUDA graphs on the first start
    exit_grace_period=60,  # seconds to finish in-flight requests on stop
    min_containers=0,  # 1 keeps a warm replica and bills four B300s around the clock
    max_containers=1,
    scaledown_window=10 * MINUTES,
)
class Server:
    @modal.enter()
    def start(self):
        self.proc = subprocess.Popen(["vllm", "serve", MODEL, *VLLM_ARGS])
        health = f"http://127.0.0.1:{PORT}/health"
        while not _up(health):
            if self.proc.poll() is not None:
                raise RuntimeError(f"vllm exited with code {self.proc.returncode}")
            time.sleep(5)

    @modal.exit()
    def stop(self):
        if hasattr(self, "proc") and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=50)


def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


@app.local_entrypoint()
def main(prompt: str = "What is 17*19? Return only the integer.", download_only: bool = False):
    """One chat request against the deployed server. Needs MODAL_PROXY_TOKEN=wk-....ws-..."""
    if download_only:
        download.remote()
        return
    token = os.environ.get("MODAL_PROXY_TOKEN")
    if not token:
        raise SystemExit("Set MODAL_PROXY_TOKEN to your proxy token id.secret first.")
    url = modal.Server.from_name("deepseek-v41-flash-starter", "Server").get_url()
    body = {"model": SERVED_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    deadline = time.monotonic() + 120 * MINUTES
    while time.monotonic() < deadline:  # retry 503 while the model starts
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                out = json.load(r)
                print(out["choices"][0]["message"]["content"], out["usage"])
                return
        except urllib.error.HTTPError as e:
            if e.code != 503:
                raise
            time.sleep(10)
    raise TimeoutError("server did not become ready")
