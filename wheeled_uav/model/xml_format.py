"""Small helpers for emitting MuJoCo XML attribute strings."""

from __future__ import annotations

from typing import Any

__all__ = ["build_xml_attributes", "format_scalar", "format_vector"]


def format_scalar(value: Any) -> str:
    if isinstance(value, str):
        return value
    return f"{float(value):g}"


def format_vector(values: list[float]) -> str:
    return " ".join(format_scalar(value) for value in values)


def build_xml_attributes(attributes: dict[str, str | None]) -> str:
    """Join name="value" pairs, dropping None and empty values."""
    return " ".join(
        f'{attribute_name}="{attribute_value}"'
        for attribute_name, attribute_value in attributes.items()
        if attribute_value is not None and attribute_value != ""
    )
