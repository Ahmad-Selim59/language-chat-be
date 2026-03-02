from litellm import acompletion
from openai import OpenAIError

from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
LANGUAGE_BUDDY_PROMPT_PATH = project_root / "src" / "system_prompts" / "language_buddy" / "language_buddy.txt"
CHAT_NAME_SUGGESTION_PROMPT_PATH = project_root / "src" / "system_prompts" / "chat_name_suggestion" / "chat_name_suggestion.txt"
INDEPTH_TRANSLATE_MESSAGE_PROMPT_PATH = project_root / "src" / "system_prompts" / "translate_message" / "indepth_translate_message.txt"
BASIC_TRANSLATE_MESSAGE_PROMPT_PATH = project_root / "src" / "system_prompts" / "translate_message" / "basic_translate_message.txt"

try:
    with open(LANGUAGE_BUDDY_PROMPT_PATH, "r", encoding="utf-8") as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    print(f"File not found: {LANGUAGE_BUDDY_PROMPT_PATH}")

try:
    with open(CHAT_NAME_SUGGESTION_PROMPT_PATH, "r", encoding="utf-8") as f:
        CHAT_NAME_SUGGESTION_PROMPT = f.read()
except FileNotFoundError:
    print(f"File not found: {CHAT_NAME_SUGGESTION_PROMPT_PATH}")

try:
    with open(INDEPTH_TRANSLATE_MESSAGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        TRANSLATE_MESSAGE_PROMPT = f.read()
except FileNotFoundError:
    print(f"File not found: {INDEPTH_TRANSLATE_MESSAGE_PROMPT_PATH}")

try:
    with open(BASIC_TRANSLATE_MESSAGE_PROMPT_PATH, "r", encoding="utf-8") as f:
        BASIC_TRANSLATE_MESSAGE_PROMPT = f.read()
except FileNotFoundError:
    print(f"File not found: {BASIC_TRANSLATE_MESSAGE_PROMPT_PATH}")


async def get_bedrock_response(
    prompt: str,
    model: str,
    previous_chat_history: list,
    settings: dict,
    temperature: float = 1,
    max_tokens: int = 500,
) -> str:
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
    system_prompt = system_prompt.replace("{{GENDER}}", settings.get("gender", "male"))
    system_prompt = system_prompt.replace("{{DIALECT}}", settings.get("dialect", "NA"))
    
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
        response = await acompletion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content
        return content
    except OpenAIError as e:
        print(f"Error calling Bedrock: {e}")
        return f"Error: {str(e)}"

async def get_chat_name_suggestion(
    prompt: str,
    model: str,
    temperature: float = 1,
    max_tokens: int = 500,
) -> str:
    """
    Function to get a chat name suggestion from the model for a new chat session

    Args:
        prompt: The text prompt to send to the model, comes from the user's first message
        model: The Bedrock model identifier (anthropic.claude-3-sonnet-20240229, amazon.titan-text-express-v1, etc.)
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate

    Returns:
        The suggested chat name
    """
    
    messages = []
    messages.append({"role": "system", "content": CHAT_NAME_SUGGESTION_PROMPT})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await acompletion(
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

async def create_translation_for_message(
    prompt: str,
    model: str,
    native_language: str,
    temperature: float = 1,
    max_tokens: int = 500,
) -> str:
    """
    Function to get a translation for a message from the model

    Args:
        prompt: The text prompt to send to the model, comes from the user's message
        model: The Bedrock model identifier (anthropic.claude-3-sonnet-20240229, amazon.titan-text-express-v1, etc.)
        temperature: Controls randomness (0.0 to 1.0)
        max_tokens: Maximum number of tokens to generate

    Returns:
        The translation of the message
    """
    system_prompt = BASIC_TRANSLATE_MESSAGE_PROMPT.replace("{{TARGET_LANGUAGE}}", native_language)

    messages = []
    messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await acompletion(
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
