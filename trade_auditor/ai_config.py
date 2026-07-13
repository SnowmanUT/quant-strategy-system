"""
Layer 2: ai_config.py

Local fallback config for ai_client.py. The environment is always checked
FIRST (ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY / AI_API_KEY,
and AI_MODEL for the model override) -- this file is only consulted if none
of those are set.

Leave these blank to rely on environment variables (recommended, especially
if this file might ever be committed to source control). Fill them in only
for local/offline convenience.
"""

# Your API key, if you're not using an environment variable.
# Anthropic keys start with "sk-ant-...", OpenAI and DeepSeek keys both
# start with "sk-..." -- set PROVIDER below to disambiguate DeepSeek.
API_KEY = ""

# Required only if API_KEY is a DeepSeek key (same "sk-..." prefix as
# OpenAI's, so it can't be auto-detected). One of: "anthropic", "openai",
# "deepseek". Leave blank for Anthropic/OpenAI keys -- prefix detection
# handles those fine.
PROVIDER = ""

# Optional model override, if you're not using the AI_MODEL environment
# variable. Leave blank to use ai_client's provider-specific default
# ("deepseek-chat" for DeepSeek).
MODEL = ""
