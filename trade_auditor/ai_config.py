"""
Layer 2: ai_config.py

Local fallback config for ai_client.py. The environment is always checked
FIRST (ANTHROPIC_API_KEY / OPENAI_API_KEY / AI_API_KEY, and AI_MODEL for the
model override) -- this file is only consulted if none of those are set.

Leave these blank to rely on environment variables (recommended, especially
if this file might ever be committed to source control). Fill them in only
for local/offline convenience.
"""

# Your API key, if you're not using an environment variable.
# Anthropic keys start with "sk-ant-...", OpenAI keys start with "sk-...".
API_KEY = ""

# Optional model override, if you're not using the AI_MODEL environment
# variable. Leave blank to use ai_client's provider-specific default.
MODEL = ""
