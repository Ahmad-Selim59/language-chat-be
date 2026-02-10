from litellm import completion
from openai import OpenAIError

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
LANGUAGE_BUDDY_PROMPT_PATH = project_root / "src" / "system_prompts" / "language_buddy" / "language_buddy.txt"

try:
    with open(LANGUAGE_BUDDY_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print(f"File not found: {LANGUAGE_BUDDY_PROMPT_PATH}")

def get_bedrock_response(
    prompt: str,
    model: str,
    previous_chat_history: list,
    settings: dict,
    temperature: float = 1,
    max_tokens: int = 500,
) -> dict:
    """
    Function to get responses from AWS Bedrock models using liteLLM

    Args:
        prompt: The text prompt to send to the model
        model: The Bedrock model identifier (anthropic.claude-3-sonnet-20240229, amazon.titan-text-express-v1, etc.)
        settings: Dictionary with targetLanguage, nativeLanguage, scriptPreference, formality
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate

    Returns:
        The complete response from the model
    """
    # Create a customized copy of the system prompt with settings
    system_prompt = SYSTEM_PROMPT.replace("{{LANGUAGE}}", settings.get("targetLanguage", "Spanish"))
    system_prompt = system_prompt.replace("{{NATIVE_LANGUAGE}}", settings.get("nativeLanguage", "English"))
    system_prompt = system_prompt.replace("{{SCRIPT_PREFERENCE}}", settings.get("scriptPreference", "target"))
    system_prompt = system_prompt.replace("{{FORMALITY_LEVEL}}", settings.get("formality", "casual"))

    # LiteLLM automatically prepends "bedrock/" to the model name
    messages = []
    messages.append({"role": "system", "content": system_prompt})

    if previous_chat_history:
        for element in previous_chat_history:
            messages.append(element)

    messages.append(
        {"role": "user", "content": prompt},
    )

    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content
    except OpenAIError as e:
        print(f"Error calling Bedrock: {e}")
        return {"error": str(e)}
