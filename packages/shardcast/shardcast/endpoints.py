"""Validated endpoint discovery metadata.

``endpoints.json`` is an optional HTTP sidecar.  Legacy clients can ignore it,
and clients that understand schema 1 can use it to select an optional UCXX
data path while retaining the HTTP base URL as a fallback.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union
from urllib.parse import urlsplit, urlunsplit


ENDPOINTS_FILENAME = "endpoints.json"
ENDPOINT_SCHEMA = 1
STREAMING_ENDPOINT_SCHEMA = 2
MAX_ENDPOINTS_JSON_SIZE = 16 * 1024
MAX_URL_SIZE = 4096
MAX_HOST_SIZE = 255
MAX_INTERFACE_SIZE = 64


class EndpointMetadataError(ValueError):
    """Raised when endpoint discovery metadata is malformed."""


@dataclass(frozen=True)
class UCXXEndpoint:
    host: str
    port: int
    interface: Optional[str] = None
    verified_rdma: bool = False


@dataclass(frozen=True)
class EndpointMetadata:
    http_base_url: str
    ucxx: Optional[UCXXEndpoint] = None
    schema: int = ENDPOINT_SCHEMA
    middle_id: Optional[str] = None
    capabilities: Tuple[str, ...] = ()

    @property
    def ucxx_host(self) -> Optional[str]:
        return None if self.ucxx is None else self.ucxx.host

    @property
    def ucxx_port(self) -> Optional[int]:
        return None if self.ucxx is None else self.ucxx.port

    @property
    def ucxx_interface(self) -> Optional[str]:
        return None if self.ucxx is None else self.ucxx.interface

    @property
    def verified_rdma(self) -> bool:
        return self.ucxx is not None and self.ucxx.verified_rdma

    def to_dict(self) -> Dict[str, Any]:
        document = {
            "schema": self.schema,
            "http": {"base_url": self.http_base_url},
        }  # type: Dict[str, Any]
        if self.ucxx is not None:
            ucxx = {
                "host": self.ucxx.host,
                "port": self.ucxx.port,
                "verified_rdma": self.ucxx.verified_rdma,
            }  # type: Dict[str, Any]
            if self.ucxx.interface is not None:
                ucxx["interface"] = self.ucxx.interface
            document["ucxx"] = ucxx
        if self.schema >= STREAMING_ENDPOINT_SCHEMA:
            document["capabilities"] = list(self.capabilities)
            if self.middle_id is not None:
                document["middle_id"] = self.middle_id
        return document

    def to_json(self) -> bytes:
        return serialize_endpoint_metadata(self)

    @classmethod
    def from_json(cls, data: Union[str, bytes, bytearray, memoryview]) -> "EndpointMetadata":
        return parse_endpoint_metadata(data)


def discovery_url(http_base_url: str) -> str:
    """Return the schema sidecar URL beneath an HTTP base URL."""
    base = _validate_http_url(http_base_url)
    parsed = urlsplit(base)
    path = parsed.path.rstrip("/") + "/" + ENDPOINTS_FILENAME
    return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


endpoint_discovery_url = discovery_url


def serialize_endpoint_metadata(metadata: EndpointMetadata) -> bytes:
    """Serialize validated metadata as deterministic, bounded UTF-8 JSON."""
    if not isinstance(metadata, EndpointMetadata):
        raise TypeError("metadata must be EndpointMetadata")
    # Validate even dataclass instances, since Python does not enforce hints.
    validated = _parse_mapping(metadata.to_dict())
    encoded = json.dumps(
        validated.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if len(encoded) > MAX_ENDPOINTS_JSON_SIZE:
        raise EndpointMetadataError("endpoint metadata exceeds maximum size")
    return encoded


def parse_endpoint_metadata(
    data: Union[str, bytes, bytearray, memoryview],
    max_size: int = MAX_ENDPOINTS_JSON_SIZE,
) -> EndpointMetadata:
    """Parse schema-1 JSON, rejecting unknown fields and oversized input."""
    if not isinstance(max_size, int) or isinstance(max_size, bool) or max_size <= 0:
        raise ValueError("max_size must be a positive integer")
    if isinstance(data, str):
        try:
            raw = data.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise EndpointMetadataError("endpoint metadata is not valid UTF-8") from exc
    else:
        try:
            raw = bytes(memoryview(data))
        except TypeError:
            raise TypeError("data must be str or bytes-like")
    if len(raw) > max_size:
        raise EndpointMetadataError("endpoint metadata exceeds maximum size")
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicates
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EndpointMetadataError("invalid endpoint metadata JSON") from exc
    if not isinstance(value, dict):
        raise EndpointMetadataError("endpoint metadata must be a JSON object")
    return _parse_mapping(value)


def write_endpoint_metadata(
    path: Union[str, os.PathLike], metadata: EndpointMetadata
) -> None:
    """Atomically replace *path* with serialized endpoint metadata."""
    target = Path(path)
    payload = serialize_endpoint_metadata(metadata)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(target.parent), prefix="." + target.name + ".",
            delete=False
        ) as handle:
            temporary = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, str(target))
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


atomic_write_endpoint_metadata = write_endpoint_metadata
EndpointDocument = EndpointMetadata
UCXXEndpointMetadata = UCXXEndpoint
parse_endpoints = parse_endpoint_metadata
serialize_endpoints = serialize_endpoint_metadata
write_endpoints = write_endpoint_metadata


def _parse_mapping(value: Mapping[str, Any]) -> EndpointMetadata:
    _require_keys(
        value, {"schema", "http"},
        {"ucxx", "middle_id", "capabilities"}, "endpoint metadata"
    )
    schema = value["schema"]
    if (
        not isinstance(schema, int)
        or isinstance(schema, bool)
        or schema not in (ENDPOINT_SCHEMA, STREAMING_ENDPOINT_SCHEMA)
    ):
        raise EndpointMetadataError("unsupported endpoint metadata schema")
    if schema == ENDPOINT_SCHEMA and (
        "middle_id" in value or "capabilities" in value
    ):
        raise EndpointMetadataError("schema 1 does not support streaming fields")

    http = value["http"]
    if not isinstance(http, dict):
        raise EndpointMetadataError("http must be an object")
    _require_keys(http, {"base_url"}, set(), "http")
    base_url = _validate_http_url(http["base_url"])

    ucxx_value = value.get("ucxx")
    ucxx = None
    if ucxx_value is not None:
        if not isinstance(ucxx_value, dict):
            raise EndpointMetadataError("ucxx must be an object")
        _require_keys(
            ucxx_value,
            {"host", "port"},
            {"interface", "verified_rdma"},
            "ucxx",
        )
        host = _validate_text(ucxx_value["host"], "ucxx host", MAX_HOST_SIZE)
        if any(character.isspace() for character in host):
            raise EndpointMetadataError("ucxx host must not contain whitespace")
        port = ucxx_value["port"]
        if (
            not isinstance(port, int)
            or isinstance(port, bool)
            or not 1 <= port <= 65535
        ):
            raise EndpointMetadataError("ucxx port must be between 1 and 65535")
        interface = ucxx_value.get("interface")
        if interface is not None:
            interface = _validate_text(
                interface, "ucxx interface", MAX_INTERFACE_SIZE
            )
            if any(character.isspace() for character in interface):
                raise EndpointMetadataError(
                    "ucxx interface must not contain whitespace"
                )
        verified = ucxx_value.get("verified_rdma", False)
        if not isinstance(verified, bool):
            raise EndpointMetadataError("verified_rdma must be boolean")
        ucxx = UCXXEndpoint(host, port, interface, verified)
    middle_id = value.get("middle_id")
    if middle_id is not None:
        middle_id = _validate_text(middle_id, "middle_id", MAX_URL_SIZE)
    capabilities_value = value.get("capabilities", [])
    if (
        not isinstance(capabilities_value, list)
        or not all(
            isinstance(item, str) and item and len(item) <= 64
            for item in capabilities_value
        )
        or len(set(capabilities_value)) != len(capabilities_value)
    ):
        raise EndpointMetadataError("capabilities must be unique strings")
    return EndpointMetadata(
        base_url, ucxx, schema, middle_id, tuple(capabilities_value)
    )


def _validate_http_url(value: Any) -> str:
    url = _validate_text(value, "http base_url", MAX_URL_SIZE)
    parsed = urlsplit(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise EndpointMetadataError("http base_url must be an absolute HTTP URL")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise EndpointMetadataError(
            "http base_url must not contain credentials, query, or fragment"
        )
    return url.rstrip("/")


def _validate_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise EndpointMetadataError("{} must be a string".format(label))
    if not value or len(value.encode("utf-8")) > maximum or "\x00" in value:
        raise EndpointMetadataError("{} is empty or too long".format(label))
    return value


def _require_keys(
    value: Mapping[str, Any], required: set, optional: set, label: str
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise EndpointMetadataError(
            "{} is missing fields: {}".format(label, ", ".join(sorted(missing)))
        )
    if unknown:
        raise EndpointMetadataError(
            "{} has unknown fields: {}".format(label, ", ".join(sorted(unknown)))
        )


def _object_without_duplicates(pairs: Any) -> Dict[str, Any]:
    value = {}  # type: Dict[str, Any]
    for key, item in pairs:
        if key in value:
            raise EndpointMetadataError("duplicate JSON field: {}".format(key))
        value[key] = item
    return value
