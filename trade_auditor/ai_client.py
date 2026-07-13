"""
Layer 2: ai_client.py

A thin, provider-routed client used only by the plain-English strategy path
(strategy_ai.py). Everything else in this app (rules.py presets, manual
rule editing, the strategy library, bars/trades in Layer 1) has zero
dependency on this module and must keep working with no API key at all.

Key resolution order:
  1. Environment variables, checked in this order: ANTHROPIC_API_KEY,
     DEEPSEEK_API_KEY, OPENAI_API_KEY, then the generic AI_API_KEY.
  2. ai_config.API_KEY, as a local fallback.

Provider routing: the named env vars (ANTHROPIC_API_KEY, DEEPSEEK_API_KEY,
OPENAI_API_KEY) each pin their provider directly -- no guessing needed.
The generic AI_API_KEY and the ai_config.py fallback route by key prefix
instead, since the provider isn't implied by the variable name:
  - starts with "sk-ant-"  -> Anthropic
  - starts with "sk-" (not "sk-ant-") -> OpenAI
  (DeepSeek keys also start with "sk-" and are indistinguishable from
  OpenAI's by prefix alone -- use DEEPSEEK_API_KEY, or set PROVIDER in
  ai_config.py, to route DeepSeek unambiguously.)

Model resolution order:
  1. AI_MODEL environment variable (overrides everything, any provider).
  2. ai_config.MODEL, as a local fallback.
  3. The provider's built-in default below.
"""

import os

import requests

DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-4o",
    "deepseek": "deepseek-chat",
}

ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"

ENV_PROVIDER_VARS = (
    ("ANTHROPIC_API_KEY", "anthropic"),
    ("DEEPSEEK_API_KEY", "deepseek"),
    ("OPENAI_API_KEY", "openai"),
)

DEFAULT_TIMEOUT_SECONDS = 30
DEFAULT_MAX_TOKENS = 1024


class AIClientError(RuntimeError):
    """Raised for any AI-client failure: no key, bad key, timeout, HTTP error. One clear line."""
    pass


def _import_ai_config():
    try:
        import ai_config
        return ai_config
    except ImportError:
        return None


def detect_provider(api_key):
    """Route by key prefix. Raises AIClientError if the format isn't recognized."""
    if api_key.startswith("sk-ant-"):
        return "anthropic"
    if api_key.startswith("sk-"):
        return "openai"
    raise AIClientError(
        "Unrecognized API key format -- expected an Anthropic key ('sk-ant-...'), "
        "an OpenAI key ('sk-...'), or a DeepSeek key set via DEEPSEEK_API_KEY."
    )


def resolve_key_and_provider():
    """
    Resolve (api_key, provider), or (None, None) if nothing is set.
    Named env vars pin their provider directly; the generic AI_API_KEY and
    the ai_config.py fallback route by key prefix (see module docstring).
    """
    for var, provider in ENV_PROVIDER_VARS:
        val = os.environ.get(var)
        if val and val.strip():
            return val.strip(), provider

    val = os.environ.get("AI_API_KEY")
    if val and val.strip():
        val = val.strip()
        return val, detect_provider(val)

    cfg = _import_ai_config()
    if cfg is not None:
        val = (getattr(cfg, "API_KEY", "") or "").strip()
        if val:
            provider = (getattr(cfg, "PROVIDER", "") or "").strip()
            return val, (provider or detect_provider(val))

    return None, None


def get_api_key():
    """Resolve the API key only. None if nothing is set."""
    return resolve_key_and_provider()[0]


def get_model(provider):
    """Resolve the model: AI_MODEL env var, then ai_config.MODEL, then the provider default."""
    override = os.environ.get("AI_MODEL")
    if override and override.strip():
        return override.strip()

    cfg = _import_ai_config()
    if cfg is not None:
        val = getattr(cfg, "MODEL", "") or ""
        if val.strip():
            return val.strip()

    return DEFAULT_MODELS[provider]


def ai_available():
    """True if a usable API key is currently resolvable (env or ai_config.py)."""
    try:
        return get_api_key() is not None
    except Exception:
        return False


def _call_anthropic(api_key, model, system_prompt, user_message, timeout, max_tokens):
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    }
    resp = requests.post(ANTHROPIC_MESSAGES_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"Anthropic API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    parts = [block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise AIClientError("Anthropic API returned no text content.")
    return text


def _call_openai(api_key, model, system_prompt, user_message, timeout, max_tokens):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"OpenAI API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        raise AIClientError("OpenAI API returned an unexpected response shape.")
    if not text:
        raise AIClientError("OpenAI API returned no text content.")
    return text


def _call_deepseek(api_key, model, system_prompt, user_message, timeout, max_tokens):
    # DeepSeek's chat completions API mirrors OpenAI's request/response shape.
    headers = {
        "Authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }
    resp = requests.post(DEEPSEEK_CHAT_URL, headers=headers, json=body, timeout=timeout)
    if resp.status_code != 200:
        raise AIClientError(f"DeepSeek API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, AttributeError):
        raise AIClientError("DeepSeek API returned an unexpected response shape.")
    if not text:
        raise AIClientError("DeepSeek API returned no text content.")
    return text


def call_ai(system_prompt, user_message, timeout=DEFAULT_TIMEOUT_SECONDS, max_tokens=DEFAULT_MAX_TOKENS):
    """
    Send a system + user message to whichever provider the resolved API key
    belongs to. Returns the raw text response. Raises AIClientError with one
    clear line on any failure (no key, bad key format, timeout, HTTP error).
    """
    api_key, provider = resolve_key_and_provider()
    if api_key is None:
        raise AIClientError(
            "No AI API key found (checked ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, "
            "OPENAI_API_KEY, AI_API_KEY, and ai_config.py). The plain-English path "
            "needs a key; presets and manual rule editing don't."
        )

    model = get_model(provider)

    try:
        if provider == "anthropic":
            return _call_anthropic(api_key, model, system_prompt, user_message, timeout, max_tokens)
        if provider == "deepseek":
            return _call_deepseek(api_key, model, system_prompt, user_message, timeout, max_tokens)
        return _call_openai(api_key, model, system_prompt, user_message, timeout, max_tokens)
    except requests.exceptions.Timeout:
        raise AIClientError(f"{provider.title()} API request timed out after {timeout}s.")
    except requests.exceptions.ConnectionError as e:
        raise AIClientError(f"Could not reach the {provider.title()} API: {e}")
    except AIClientError:
        raise
    except Exception as e:
        raise AIClientError(f"Unexpected {provider.title()} API client error: {e}")


if __name__ == "__main__":
    key, provider = resolve_key_and_provider()
    if key is None:
        print("No AI key configured (env or ai_config.py). Plain-English path would be greyed out.")
    else:
        print(f"Provider: {provider}, model: {get_model(provider)}")
