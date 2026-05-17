import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_SAMPLE_AUDIO_URL = "https://static.deepgram.com/examples/Bueller-Life-moves-pretty-fast.wav"


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if key and key not in os.environ:
            os.environ[key] = value


def download_audio(audio_url: str) -> bytes:
    request = Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def transcribe_audio(api_key: str, audio_bytes: bytes) -> dict:
    query = urlencode(
        {
            "model": os.getenv("DEEPGRAM_MODEL", "nova-2"),
            "smart_format": "true",
            "punctuate": "true",
        }
    )

    request = Request(
        f"https://api.deepgram.com/v1/listen?{query}",
        data=audio_bytes,
        headers={
            "Authorization": f"Token {api_key}",
            "Content-Type": "audio/wav",
            "Accept": "application/json",
        },
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def extract_transcript(response_data: dict) -> str:
    try:
        return response_data["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
    except (KeyError, IndexError, TypeError):
        return ""


def main() -> int:
    load_env_file(Path(__file__).with_name(".env"))

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Missing DEEPGRAM_API_KEY in environment or .env", file=sys.stderr)
        return 1

    audio_url = os.getenv("DEEPGRAM_TEST_AUDIO_URL", DEFAULT_SAMPLE_AUDIO_URL)

    try:
        audio_bytes = download_audio(audio_url)
        response_data = transcribe_audio(api_key, audio_bytes)
        transcript = extract_transcript(response_data)

        if not transcript:
            print("Deepgram transcription test failed", file=sys.stderr)
            print("No transcript was returned.", file=sys.stderr)
            print(json.dumps(response_data, indent=2), file=sys.stderr)
            return 1

        print("Deepgram transcription test passed")
        print(f"Sample URL: {audio_url}")
        print(f"Transcript: {transcript}")
        return 0
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print("Deepgram transcription test failed", file=sys.stderr)
        print(f"HTTP status: {error.code}", file=sys.stderr)
        if error_body:
            print(error_body, file=sys.stderr)
        return 1
    except URLError as error:
        print("Deepgram transcription test failed", file=sys.stderr)
        print(f"Network error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())