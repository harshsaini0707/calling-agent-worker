import os
import sys


REQUIRED_TELEPHONY_KEYS = [
    "LIVEKIT_URL",
    "LIVEKIT_API_KEY",
    "LIVEKIT_API_SECRET",
    "OPENAI_API_KEY",
    "SARVAM_API_KEY",
    "OUTBOUND_TRUNK_ID",
    "BACKEND_WEBHOOK_URL",
    "CALL_SCREENING_WEBHOOK_SECRET",
]


def main():
    errors = []

    for key in REQUIRED_TELEPHONY_KEYS:
        if not str(os.getenv(key, "")).strip():
            errors.append(f"Missing required telephony env: {key}")

    webhook_url = str(os.getenv("BACKEND_WEBHOOK_URL", "")).strip()
    if webhook_url.startswith("http://localhost"):
        errors.append(
            "BACKEND_WEBHOOK_URL points to localhost. In Docker-local mode use host.docker.internal or a remote URL."
        )

    if errors:
        print("Telephony local preflight failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Telephony local preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
