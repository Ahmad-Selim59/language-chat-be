import pytest
import os
from src.gcp import transcribe_audio

@pytest.mark.asyncio
async def test_deepgram_transcribe_audio():
    # Use a small valid RIFF WAVE header with no actual sound data
    # to test API authorization and structure without hitting size limits
    # Just to confirm we don't get 400 Bad Request or 401 Unauthorized
    b64_audio = "UklGRiQAAABXQVZFZm10IBAAAAABAAEAQB8AAEF/AAABAAgAZGF0YQAAAAA="
    
    try:
        result = await transcribe_audio(b64_audio)
        # Deepgram transcribes empty data as an empty string, which is correct.
        # As long as it doesn't throw a HTTPStatusError, the integration is successful.
        assert isinstance(result, str)
    except Exception as e:
        pytest.fail(f"transcribe_audio raised an exception: {e}")
