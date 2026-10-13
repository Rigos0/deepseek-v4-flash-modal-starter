"""DeepSeek-V4-Flash-0731 on one Modal B300 with vLLM.

    modal secret create huggingface HF_TOKEN=hf_...   # once
    modal deploy deepseek_v4_flash.py           # publish the endpoint
    modal run deepseek_v4_flash.py              # send one test request
    modal app stop deepseek-v4-flash-starter                  # stop paying

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

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"  # 2026-07-31 release, 167 GB
SERVED_NAME = "deepseek-v4-flash"
# vllm/vllm-openai:v0.31.0 (2026-10-04), CUDA 13.0. Pinned by digest so a rebuild
# cannot silently change the engine.
VLLM_IMAGE = "vllm/vllm-openai@sha256:c1c9f6fd5c109ba7f0546a59f5b2f15fb87f64c77782e90a27b648b42a8e67c3"
GPU = "B300:1"
PORT = 8000
MINUTES = 60

# Model-specific settings from the vLLM recipe; tune against your workload.
VLLM_ARGS = [
    "--revision", REVISION,
    "--served-model-name", SERVED_NAME,
    "--trust-remote-code",
    "--tokenizer-mode", "deepseek_v4",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--tensor-parallel-size", "1",
    "--kv-cache-dtype", "fp8",
    "--block-size", "256",
    "--gpu-memory-utilization", "0.92",
    "--max-model-len", "262144",
    "--max-num-seqs", "64",
    "--max-num-batched-tokens", "8192",
    "--attention-config", '{"indexer_kv_dtype":"mxfp4"}',
    "--speculative-config", '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}',
    "--reasoning-parser", "deepseek_v4",
    "--tool-call-parser", "deepseek_v4", "--enable-auto-tool-choice",
]

# The vLLM image ships its own Python for vLLM but no plain `python` on PATH, so Modal
# gets a separate interpreter for its own runtime (add_python).
image = (
    modal.Image.from_registry(VLLM_IMAGE, add_python="3.12")
    .entrypoint([])
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "VLLM_ENGINE_READY_TIMEOUT_S": "3600"})
)

app = modal.App("deepseek-v4-flash-starter")


@app.server(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=96 * 1024,  # MB; weights stream through host memory while loading
    port=PORT,
    volumes={
        "/root/.cache/huggingface": modal.Volume.from_name("deepseek-flash-hf", create_if_missing=True),
        "/root/.cache/vllm": modal.Volume.from_name("deepseek-flash-compile", create_if_missing=True),
    },
    secrets=[modal.Secret.from_name("huggingface")],
    startup_timeout=60 * MINUTES,  # first start: 167 GB download, then kernel JIT and CUDA graphs
    exit_grace_period=60,  # seconds to finish in-flight requests on stop
    min_containers=0,  # 1 keeps a warm replica and bills a B300 around the clock
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
        self.proc.terminate()
        self.proc.wait()


def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


@app.local_entrypoint()
def main(prompt: str = "What is 17*19? Return only the integer."):
    """One chat request against the deployed server. Needs MODAL_PROXY_TOKEN=wk-....ws-..."""
    token = os.environ.get("MODAL_PROXY_TOKEN")
    if not token:
        raise SystemExit("Set MODAL_PROXY_TOKEN to your proxy token id.secret first.")
    url = modal.Server.from_name("deepseek-v4-flash-starter", "Server").get_url()
    body = {"model": SERVED_NAME, "messages": [{"role": "user", "content": prompt}], "max_tokens": 64,
            "chat_template_kwargs": {"thinking": False}}
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"},
    )
    deadline = time.monotonic() + 60 * MINUTES
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
