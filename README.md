# DeepSeek V4 Flash on Modal

Minimal companion to the Era Compute walkthrough: one B300, vLLM, an authenticated OpenAI-compatible endpoint, and an OpenCode config.

The server is in [`deepseek_v4_flash.py`](deepseek_v4_flash.py). Model and engine versions are pinned. **GPU execution and the complete OpenCode tool loop have not yet been validated for this example.**

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
modal deploy deepseek_v4_flash.py
```

Save the HTTPS URL printed by Modal. The first request starts the model download and loading process.

## 3. Send a request

```sh
modal workspace proxy-tokens create
export MODAL_PROXY_TOKEN="wk-YOUR_ID.ws-YOUR_SECRET"
modal run deepseek_v4_flash.py
```

Use the proxy token ID and secret returned by the first command, joined with a dot. The test sends one chat request and retries HTTP 503 while the model starts. Keep both tokens out of Git.

## 4. Connect OpenCode

Copy [`opencode.json`](opencode.json) into your coding project, or merge its provider into your existing config. Replace the `baseURL` placeholder with your Modal URL followed by `/v1`.

From the terminal where `MODAL_PROXY_TOKEN` is set, launch `opencode` and select `modal/deepseek-v4-flash`. The agent runs on your laptop; model requests go to Modal.

## Stop

```sh
modal app stop deepseek-v4-flash-starter
```

The example allows one GPU container and scales to zero after 10 idle minutes. Startup and the idle window are billed. After shutdown, the next request triggers a cold start and receives 503 until ready; rerun the test before opening OpenCode. The two persistent cache volumes remain after stopping the app.

[Modal setup](https://modal.com/docs/guide) · [Modal Servers](https://modal.com/docs/guide/servers) · [vLLM recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash) · [Model](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)
