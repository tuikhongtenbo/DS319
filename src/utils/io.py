"""
I/O utilities for loading and saving data (e.g., JSON/JSONL).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def load_json(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Loads a JSON file."""
    filepath = Path(filepath)
    assert filepath.exists(), f"File not found: {filepath}"
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves data to a JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads a JSONL file."""
    filepath = Path(filepath)
    assert filepath.exists(), f"File not found: {filepath}"
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))
    return data


def save_jsonl(data: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Saves data to a JSONL file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
