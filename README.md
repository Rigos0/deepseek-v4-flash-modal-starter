# DeepSeek V4.1 Flash on Modal

Minimal companion to the Era Compute walkthrough: four B300s, vLLM, an authenticated OpenAI-compatible endpoint, and an OpenCode config.

The server is in [`deepseek_v41_flash.py`](deepseek_v41_flash.py). Model and engine versions are pinned. **V4.1 GPU execution and the complete OpenCode tool loop are being validated; no performance claim yet.**

This uses the official V4.1 checkpoint, a pinned vLLM nightly, and tensor parallelism across four B300s. It serves text only for coding, with a 256K context cap and five-token DSpark drafting. Modal credits contribute toward usage; this four-GPU configuration costs about $31/hour including the requested CPU and RAM, so the first setup can exceed the free credits.

## 1. Set up Modal

Create an account at [modal.com](https://modal.com/). GPU use requires a payment method; credits apply toward usage. On your laptop, with Python 3.10+:

```sh
git clone https://github.com/Rigos0/deepseek-v4-flash-modal-starter.git
cd deepseek-v4-flash-modal-starter
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

`modal setup` opens your browser to connect the CLI to your account.

## 2. Deploy

Create a [Hugging Face read token](https://huggingface.co/settings/tokens), then replace the placeholder below:

```sh
modal secret create huggingface HF_TOKEN=hf_YOUR_TOKEN
modal run deepseek_v41_flash.py --download-only
modal deploy deepseek_v41_flash.py
```

The download command caches the roughly 511 GB checkpoint on a CPU container. Save the HTTPS URL printed by deploy. The first request starts four B300s, loads the weights, and prepares the kernels.

## 3. Send a request

```sh
modal workspace proxy-tokens create
export MODAL_PROXY_TOKEN="wk-YOUR_ID.ws-YOUR_SECRET"
modal run deepseek_v41_flash.py
```

Use the proxy token ID and secret returned by the first command, joined with a dot. The test sends one chat request and retries HTTP 503 while the model starts. Keep both tokens out of Git.

## 4. Connect OpenCode

Copy [`opencode.json`](opencode.json) into your coding project, or merge its provider into your existing config. Replace the `baseURL` placeholder with your Modal URL followed by `/v1`.

From the terminal where `MODAL_PROXY_TOKEN` is set, launch `opencode` and select `modal/deepseek-v4.1-flash`. The agent runs on your laptop; model requests go to Modal.

## Stop

```sh
modal app stop deepseek-v41-flash-starter
```

The example allows one container with four GPUs and scales to zero after 10 idle minutes. GPU startup and the idle window are billed. First-time kernel preparation can take a while; the test waits up to two hours. After shutdown, the next request triggers a cold start and receives 503 until ready; rerun the test before opening OpenCode. The two persistent cache volumes remain after stopping the app.

[Modal setup](https://modal.com/docs/guide) · [Modal Servers](https://modal.com/docs/guide/servers) · [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4.1-Flash) · [Model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
