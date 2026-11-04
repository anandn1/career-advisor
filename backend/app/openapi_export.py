"""Write the API's OpenAPI schema as JSON (for generating frontend types).

Run from backend/:  uv run python -m app.openapi_export ../frontend/openapi.json
"""

import json
import sys
from pathlib import Path

from app.main import create_app


def main() -> None:
    schema = json.dumps(create_app().openapi(), indent=2, sort_keys=True) + "\n"
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(schema, encoding="utf-8")
    else:
        sys.stdout.write(schema)


if __name__ == "__main__":
    main()
