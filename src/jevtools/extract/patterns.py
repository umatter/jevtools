"""Pattern extractors: ``email url uuid ipv4 code regex:<re>`` (spec §3.2 ``extract``, §4.2.6).

``code`` matches identifier-like tokens: a letter plus a digit or ``_`` (``DNA123``, ``SKU-4411``, ``v2.3.1``,
``x^2``), or a file name with an extension (``notes_old.txt``, ``report.pdf``). Code mentions are weak: they never
claim other mentions, and a more specific reading (``3pm``, ``5kg``) claims them.
"""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable

from jevtools.extract.base import Mention, SourceText

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'`]+", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_TRAILING = ".,;:!?)]}"
_WORDISH = re.compile(r"[^\s,;:!?()\[\]{}<>\"'`]+")
_FILE_NAME = re.compile(r"[^\W_][\w\-]+\.[A-Za-z0-9]{1,5}")
CODE_MAX = 64

PATTERNS: dict[str, re.Pattern[str]] = {"email": EMAIL_RE, "url": URL_RE, "uuid": UUID_RE, "ipv4": IPV4_RE}


def _valid_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text)
    except ValueError:
        return False
    return True


def normalize_email(address: str) -> str:
    """Email normalizer (§4.3): lowercase the domain, keep the local part verbatim."""
    local, _, domain = address.rpartition("@")
    return f"{local}@{domain.lower()}" if local else address


def extract(source: SourceText, regexes: Iterable[str] = ()) -> list[Mention]:
    """Emails, URLs, UUIDs, IPv4 addresses and declared ``regex:<re>`` matches in one text."""
    out: list[Mention] = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(source.text):
            text = match.group()
            end = match.end()
            if kind == "url":
                stripped = text.rstrip(_TRAILING)
                end -= len(text) - len(stripped)
                text = stripped
            if kind == "ipv4" and not _valid_ipv4(text):
                continue
            value = normalize_email(text) if kind == "email" else text
            out.append(Mention(kind, text, (match.start(), end), value, None, source.channel, source.ref, kind))
    out += _codes(source)
    for expr in regexes:
        try:
            pattern = re.compile(expr)
        except re.error:
            continue
        for match in pattern.finditer(source.text):
            if match.group():
                out.append(
                    Mention(
                        "regex",
                        match.group(),
                        match.span(),
                        match.group(),
                        None,
                        source.channel,
                        source.ref,
                        "regex",
                        attrs={"regex": expr},
                    )
                )
    return out


def is_code(token: str) -> bool:
    """Whether ``token`` looks like an identifier or a file name (see the module docstring)."""
    if not 2 <= len(token) <= CODE_MAX or "@" in token or URL_RE.match(token):
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return any(ch.isdigit() or ch == "_" for ch in token) or _FILE_NAME.fullmatch(token) is not None


def _codes(source: SourceText) -> list[Mention]:
    out: list[Mention] = []
    for match in _WORDISH.finditer(source.text):
        text = match.group().rstrip(".-/")
        text = text[:-2] if text.endswith(("'s", "’s")) else text
        if is_code(text):
            span = (match.start(), match.start() + len(text))
            out.append(Mention("code", text, span, text, None, source.channel, source.ref, "code",
                               attrs={"claims": False}))  # fmt: skip
    return out


__all__ = ["EMAIL_RE", "IPV4_RE", "PATTERNS", "URL_RE", "UUID_RE", "extract", "is_code", "normalize_email"]
