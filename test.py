import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


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


def test_deepgram_api() -> int:
    load_env_file(Path(__file__).with_name(".env"))

    api_key = os.getenv("DEEPGRAM_API_KEY")
    if not api_key:
        print("Missing DEEPGRAM_API_KEY in environment or .env", file=sys.stderr)
        return 1

    request = Request(
        "https://api.deepgram.com/v1/projects",
        headers={
            "Authorization": f"Token {api_key}",
            "Accept": "application/json",
        },
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            body = response.read().decode("utf-8", errors="replace")
            print("Deepgram API test passed")
            print(f"HTTP status: {response.status}")
            print(body)
            return 0
    except HTTPError as error:
        error_body = error.read().decode("utf-8", errors="replace")
        print("Deepgram API test failed", file=sys.stderr)
        print(f"HTTP status: {error.code}", file=sys.stderr)
        if error_body:
            print(error_body, file=sys.stderr)
        return 1
    except URLError as error:
        print("Deepgram API test failed", file=sys.stderr)
        print(f"Network error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(test_deepgram_api())