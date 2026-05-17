"""Markdown decision block loader for pipeline scripts."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml


FENCED_BLOCK_RE = re.compile(r"```([^\n`]*)\n(.*?)\n```", re.DOTALL)


def load_decision_block(markdown_path: str | None, block_name: str) -> dict[str, Any]:
    """Load a named YAML block from a Markdown file.

    A valid block looks like:

    ```yaml pipeline_decisions
    key: value
    ```
    """
    if not markdown_path:
        return {}

    path = Path(markdown_path)
    if not path.exists():
        raise FileNotFoundError(f"decision Markdown 파일을 찾을 수 없습니다: {path}")

    text = path.read_text(encoding="utf-8")
    for info_string, body in FENCED_BLOCK_RE.findall(text):
        info_tokens = {token.strip() for token in info_string.split() if token.strip()}
        if block_name not in info_tokens:
            continue
        payload = yaml.safe_load(body) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"{path}의 {block_name} block은 YAML object 형태여야 합니다.")
        return payload

    raise ValueError(
        f"{path}에서 `{block_name}` YAML block을 찾을 수 없습니다. "
        f"예: ```yaml {block_name}"
    )
