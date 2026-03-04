#!/usr/bin/env python
"""
Test script to verify all LLM providers work with the refactored code.

Tests the UnifiedLLMClient with each supported provider.
"""

import sys
import os

# Add the src directory to path so we import from local code
src_path = '/Users/chrissoria/Documents/Research/cat-ademic/src'
sys.path.insert(0, src_path)

# Clear any cached modules
modules_to_remove = [key for key in sys.modules.keys() if key.startswith('catademic')]
for mod in modules_to_remove:
    del sys.modules[mod]

from dotenv import load_dotenv

# Load API keys
load_dotenv('/Users/chrissoria/Documents/Research/Categorization_AI_experiments/.env')

# Get all API keys
api_keys = {
    "openai": os.getenv("OPENAI_API_KEY"),
    "anthropic": os.getenv("ANTHROPIC_API_KEY"),
    "google": os.getenv("GOOGLE_API_KEY"),
    "mistral": os.getenv("MISTRAL_API_KEY"),
    "xai": os.getenv("XAI_API_KEY"),
    "huggingface": os.getenv("HUGGINGFACE_API_KEY"),
    "perplexity": os.getenv("PERPLEXITY_API_KEY"),
}

print("=" * 70)
print("All Providers Test - UnifiedLLMClient")
print("=" * 70)
print()

# Print API key status
print("API Keys Status:")
for provider, key in api_keys.items():
    status = "Found" if key else "MISSING"
    print(f"  {provider}: {status}")
print()

# Import the client
from catademic.text_functions import UnifiedLLMClient

# Test configurations for each provider
# Using small/fast models to minimize cost and time
test_configs = [
    {
        "name": "OpenAI",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "api_key": api_keys["openai"],
    },
    {
        "name": "Anthropic",
        "provider": "anthropic",
        "model": "claude-3-haiku-20240307",
        "api_key": api_keys["anthropic"],
    },
    {
        "name": "Google",
        "provider": "google",
        "model": "gemini-2.0-flash",
        "api_key": api_keys["google"],
    },
    {
        "name": "Mistral",
        "provider": "mistral",
        "model": "mistral-small-latest",
        "api_key": api_keys["mistral"],
    },
    {
        "name": "xAI",
        "provider": "xai",
        "model": "grok-3-mini-fast",
        "api_key": api_keys["xai"],
    },
    {
        "name": "Perplexity",
        "provider": "perplexity",
        "model": "sonar",
        "api_key": api_keys["perplexity"],
    },
    {
        "name": "HuggingFace (Qwen - generic endpoint)",
        "provider": "huggingface",
        "model": "Qwen/Qwen2.5-72B-Instruct",
        "api_key": api_keys["huggingface"],
    },
    {
        "name": "HuggingFace (Llama 4 - together endpoint, auto-detect)",
        "provider": "huggingface",
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "api_key": api_keys["huggingface"],
    },
    {
        "name": "HuggingFace-Together (explicit)",
        "provider": "huggingface-together",
        "model": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "api_key": api_keys["huggingface"],
    },
]

# Simple test message
test_messages = [
    {"role": "user", "content": "Reply with exactly one word: hello"}
]

results = []

for i, config in enumerate(test_configs, 1):
    print(f"\n{'-' * 70}")
    print(f"Test {i}/{len(test_configs)}: {config['name']}")
    print(f"{'-' * 70}")
    print(f"  Provider: {config['provider']}")
    print(f"  Model: {config['model']}")

    if not config['api_key']:
        print(f"  Result: SKIPPED - No API key")
        results.append({
            "name": config['name'],
            "provider": config['provider'],
            "model": config['model'],
            "success": None,
            "skipped": True,
            "error": "No API key",
        })
        continue

    print(f"  Testing...")

    try:
        client = UnifiedLLMClient(
            provider=config['provider'],
            api_key=config['api_key'],
            model=config['model']
        )

        response, error = client.complete(
            messages=test_messages,
            creativity=0.1,
            force_json=False,  # Simple text response
        )

        if error:
            print(f"  Result: FAIL - {error}")
            results.append({
                "name": config['name'],
                "provider": config['provider'],
                "model": config['model'],
                "success": False,
                "skipped": False,
                "error": error,
            })
        else:
            # Truncate response for display
            display_response = response[:50] + "..." if len(response) > 50 else response
            display_response = display_response.replace('\n', ' ')
            print(f"  Response: {display_response}")
            print(f"  Result: PASS")
            results.append({
                "name": config['name'],
                "provider": config['provider'],
                "model": config['model'],
                "success": True,
                "skipped": False,
                "response": response,
            })

    except Exception as e:
        print(f"  Result: ERROR - {str(e)}")
        results.append({
            "name": config['name'],
            "provider": config['provider'],
            "model": config['model'],
            "success": False,
            "skipped": False,
            "error": str(e),
        })

# Summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

passed = sum(1 for r in results if r.get('success') is True)
failed = sum(1 for r in results if r.get('success') is False)
skipped = sum(1 for r in results if r.get('skipped') is True)

print(f"  Passed:  {passed}")
print(f"  Failed:  {failed}")
print(f"  Skipped: {skipped}")
print(f"  Total:   {len(results)}")

if failed > 0:
    print(f"\nFailed tests:")
    for r in results:
        if r.get('success') is False:
            print(f"  - {r['name']}: {r.get('error', 'Unknown error')}")

if skipped > 0:
    print(f"\nSkipped tests (no API key):")
    for r in results:
        if r.get('skipped'):
            print(f"  - {r['name']}")

print()

# Exit with appropriate code
sys.exit(1 if failed > 0 else 0)
