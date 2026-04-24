"""Minimal DICOMweb client (QIDO-RS + WADO-RS).

Supports:
  - QIDO-RS: search studies, search series in a study
  - WADO-RS: retrieve instances as application/dicom (multipart) or one-shot
    per-series retrieve which most servers implement

Transport: HTTPS with optional bearer token or basic auth. No dcm4che / no
pydicom-networking dep — just `requests` and the multipart parser in stdlib.

Usage:

    client = DicomWebClient("https://dicomweb.example/dicom-web", token="…")
    studies = client.search_studies({"PatientID": "12345"})
    series = client.search_series(studies[0]["0020000D"]["Value"][0])
    paths = client.retrieve_series(study_uid, series_uid, dest_dir)
    # then: load_path(dest_dir)
"""

from __future__ import annotations

import email
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlencode

import requests


_BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.IGNORECASE)


@dataclass
class DicomWebClient:
    base_url: str
    token: str | None = None
    username: str | None = None
    password: str | None = None
    verify_tls: bool = True
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    # --- internals ---

    def _auth_headers(self, accept: str) -> dict[str, str]:
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _auth_tuple(self) -> tuple[str, str] | None:
        if self.username and self.password is not None:
            return (self.username, self.password)
        return None

    def _get(self, path: str, params: dict[str, str] | None = None,
             accept: str = "application/dicom+json") -> requests.Response:
        url = f"{self.base_url}/{path.lstrip('/')}"
        if params:
            url = f"{url}?{urlencode(params)}"
        resp = requests.get(
            url,
            headers=self._auth_headers(accept),
            auth=self._auth_tuple(),
            verify=self.verify_tls,
            timeout=self.timeout,
            stream=accept.startswith("multipart/"),
        )
        resp.raise_for_status()
        return resp

    # --- QIDO-RS ---

    def search_studies(self, filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        resp = self._get("studies", params=filters)
        return resp.json() if resp.content else []

    def search_series(self, study_uid: str,
                      filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
        resp = self._get(
            f"studies/{quote(study_uid, safe='')}/series", params=filters
        )
        return resp.json() if resp.content else []

    def search_instances(self, study_uid: str, series_uid: str) -> list[dict[str, Any]]:
        resp = self._get(
            f"studies/{quote(study_uid, safe='')}"
            f"/series/{quote(series_uid, safe='')}/instances"
        )
        return resp.json() if resp.content else []

    # --- WADO-RS ---

    def retrieve_series(self, study_uid: str, series_uid: str,
                        dest_dir: Path) -> list[Path]:
        """Fetch all instances of a series as multipart/related; write to disk."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        resp = self._get(
            f"studies/{quote(study_uid, safe='')}"
            f"/series/{quote(series_uid, safe='')}",
            accept="multipart/related; type=application/dicom",
        )
        return _write_multipart_parts(resp, dest_dir)


def _write_multipart_parts(resp: requests.Response, dest_dir: Path) -> list[Path]:
    content_type = resp.headers.get("Content-Type", "")
    m = _BOUNDARY_RE.search(content_type)
    if not m:
        raise RuntimeError(f"Response is not multipart: {content_type!r}")
    boundary = m.group(1)

    body = resp.content
    # Reconstruct a parseable multipart envelope
    header = f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n".encode()
    message = email.message_from_bytes(header + body)
    paths: list[Path] = []
    index = 0
    for part in message.walk():
        if part.is_multipart():
            continue
        payload = part.get_payload(decode=True)
        if not payload:
            continue
        path = dest_dir / f"instance_{index:04d}_{uuid.uuid4().hex[:6]}.dcm"
        path.write_bytes(payload)
        paths.append(path)
        index += 1
    if not paths:
        raise RuntimeError("No DICOM parts returned by WADO-RS.")
    return paths


def build_dicomweb_url(base: str, *segments: str, **params: str) -> str:
    """Helper used by tests; also useful in debugging."""
    url = base.rstrip("/") + "/" + "/".join(quote(s, safe="") for s in segments)
    if params:
        url += "?" + urlencode(params)
    return url
