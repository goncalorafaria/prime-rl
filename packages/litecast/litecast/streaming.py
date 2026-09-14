"""Metadata and deterministic placement for replicated shard streaming."""

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import blake3


MANIFEST_FILENAME = "manifest.json"
LEGACY_MANIFEST_SCHEMA = 1
MANIFEST_SCHEMA = 2
MAX_MANIFEST_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ShardManifest:
    name: str
    total_size: int
    shard_size: int
    shard_count: int
    checksum: str
    shard_checksums: Tuple[str, ...]
    members: Tuple[str, ...] = ()
    replication_factor: int = 1
    merkle_root: Optional[str] = None
    member_profiles: Tuple[
        Tuple[str, float, float, float, str, str], ...
    ] = ()
    schema: int = MANIFEST_SCHEMA

    def shard_length(self, index: int) -> int:
        if index < 0 or index >= self.shard_count:
            raise IndexError(index)
        return min(self.shard_size, self.total_size - index * self.shard_size)

    def to_dict(self) -> Mapping[str, Any]:
        document = {
            "schema": self.schema,
            "name": self.name,
            "total_size": self.total_size,
            "shard_size": self.shard_size,
            "shard_count": self.shard_count,
            "checksum": self.checksum,
            "shard_checksums": list(self.shard_checksums),
            "members": list(self.members),
            "replication_factor": self.replication_factor,
        }
        if self.schema >= MANIFEST_SCHEMA:
            document["merkle_root"] = self.merkle_root
            document["member_profiles"] = {
                member: {
                    "bandwidth": bandwidth,
                    "capacity": capacity,
                    "load": load,
                    "failure_domain": failure_domain,
                    "locality": locality,
                }
                for (
                    member,
                    bandwidth,
                    capacity,
                    load,
                    failure_domain,
                    locality,
                ) in self.member_profiles
            }
        return document

    @property
    def profiles(self) -> Dict[str, Mapping[str, Any]]:
        return {
            member: {
                "bandwidth": bandwidth,
                "capacity": capacity,
                "load": load,
                "failure_domain": failure_domain,
                "locality": locality,
            }
            for (
                member,
                bandwidth,
                capacity,
                load,
                failure_domain,
                locality,
            ) in self.member_profiles
        }

    def to_bytes(self) -> bytes:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        if len(payload) > MAX_MANIFEST_BYTES:
            raise ValueError("shard manifest exceeds size limit")
        return payload

    @classmethod
    def from_bytes(cls, data: Any) -> "ShardManifest":
        raw = bytes(memoryview(data))
        if len(raw) > MAX_MANIFEST_BYTES:
            raise ValueError("shard manifest exceeds size limit")
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid shard manifest") from exc
        if not isinstance(value, dict):
            raise ValueError("shard manifest must be an object")
        schema = value.get("schema")
        required = {
            "schema", "name", "total_size", "shard_size", "shard_count",
            "checksum", "shard_checksums",
            "members", "replication_factor",
        }
        if schema == MANIFEST_SCHEMA:
            required |= {"merkle_root", "member_profiles"}
        if (
            schema not in (LEGACY_MANIFEST_SCHEMA, MANIFEST_SCHEMA)
            or set(value) != required
        ):
            raise ValueError("unsupported shard manifest schema")
        name = value["name"]
        total_size = _integer(value["total_size"], "total_size", 0)
        shard_size = _integer(value["shard_size"], "shard_size", 1)
        shard_count = _integer(value["shard_count"], "shard_count", 0)
        expected_count = (
            (total_size + shard_size - 1) // shard_size if total_size else 0
        )
        checksums = value["shard_checksums"]
        members_value = value["members"]
        replication_factor = value["replication_factor"]
        root = value.get("merkle_root")
        profiles_value = value.get("member_profiles", {})
        try:
            profiles = (
                canonical_member_profiles(members_value, profiles_value)
                if members_value
                else ()
            )
        except (TypeError, ValueError):
            raise ValueError("inconsistent shard manifest")
        if (
            not isinstance(name, str)
            or not name
            or shard_count != expected_count
            or not isinstance(checksums, list)
            or len(checksums) != shard_count
            or not _checksum(value["checksum"])
            or not all(_checksum(item) for item in checksums)
            or not isinstance(members_value, list)
            or not all(isinstance(item, str) for item in members_value)
            or (
                members_value
                and tuple(members_value) != canonical_members(members_value)
            )
            or not isinstance(replication_factor, int)
            or isinstance(replication_factor, bool)
            or replication_factor < 1
            or (
                members_value
                and replication_factor > len(members_value)
            )
            or (
                schema == MANIFEST_SCHEMA
                and (
                    not _checksum(root)
                    or root != merkle_root(tuple(checksums))
                    or set(profiles_value) != set(members_value)
                )
            )
        ):
            raise ValueError("inconsistent shard manifest")
        return cls(
            name=name,
            total_size=total_size,
            shard_size=shard_size,
            shard_count=shard_count,
            checksum=value["checksum"],
            shard_checksums=tuple(checksums),
            members=tuple(members_value),
            replication_factor=replication_factor,
            merkle_root=root,
            member_profiles=profiles,
            schema=schema,
        )


def build_manifest(name: str, data: Any, shard_size: int) -> ShardManifest:
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if (
        not isinstance(shard_size, int)
        or isinstance(shard_size, bool)
        or shard_size <= 0
    ):
        raise ValueError("shard_size must be positive")
    view = memoryview(data)
    total_size = len(view)
    shard_count = (total_size + shard_size - 1) // shard_size if total_size else 0
    shard_checksums = tuple(
        blake3.blake3(view[index * shard_size:min((index + 1) * shard_size, total_size)]).hexdigest()
        for index in range(shard_count)
    )
    return ShardManifest(
        name=name,
        total_size=total_size,
        shard_size=shard_size,
        shard_count=shard_count,
        checksum=blake3.blake3(view).hexdigest(),
        shard_checksums=shard_checksums,
        merkle_root=merkle_root(shard_checksums),
    )


def merkle_root(shard_checksums: Iterable[str]) -> str:
    """Return a domain-separated ordered Merkle root for shard digests."""
    level = [
        blake3.blake3(
            b"litecast-leaf\0" + bytes.fromhex(checksum)
        ).digest()
        for checksum in shard_checksums
    ]
    if not level:
        return blake3.blake3(b"litecast-empty\0").hexdigest()
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [
            blake3.blake3(
                b"litecast-node\0" + level[index] + level[index + 1]
            ).digest()
            for index in range(0, len(level), 2)
        ]
    return level[0].hex()


def canonical_members(members: Iterable[str]) -> Tuple[str, ...]:
    result = tuple(str(item).rstrip("/") for item in members)
    if not result or any(not item for item in result) or len(set(result)) != len(result):
        raise ValueError("middle membership must contain unique non-empty URLs")
    return result


def shard_owners(
    version: str,
    shard_index: int,
    members: Iterable[str],
    replication_factor: int = 2,
    member_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Tuple[str, ...]:
    """Return weighted rendezvous owners with failure-domain diversity."""
    canonical = canonical_members(members)
    if not isinstance(replication_factor, int) or isinstance(replication_factor, bool):
        raise TypeError("replication_factor must be int")
    if replication_factor < 1:
        raise ValueError("replication_factor must be positive")
    count = min(replication_factor, len(canonical))
    profiles = canonical_member_profiles(canonical, member_profiles or {})
    profile_map = {
        item[0]: item[1:] for item in profiles
    }
    ranked = sorted(
        canonical,
        key=lambda member: _weighted_score(
            version, shard_index, member, profile_map[member]
        ),
        reverse=True,
    )
    selected = []
    domains = set()
    for member in ranked:
        domain = profile_map[member][3]
        if domain and domain in domains:
            continue
        selected.append(member)
        if domain:
            domains.add(domain)
        if len(selected) == count:
            return tuple(selected)
    for member in ranked:
        if member not in selected:
            selected.append(member)
        if len(selected) == count:
            break
    return tuple(selected)


def canonical_member_profiles(
    members: Iterable[str],
    profiles: Mapping[str, Mapping[str, Any]],
) -> Tuple[Tuple[str, float, float, float, str, str], ...]:
    canonical = canonical_members(members)
    if not isinstance(profiles, Mapping):
        raise TypeError("member_profiles must be a mapping")
    unknown = set(profiles) - set(canonical)
    if unknown:
        raise ValueError("member profile does not belong to membership")
    result = []
    for member in canonical:
        profile = profiles.get(member, {})
        if not isinstance(profile, Mapping):
            raise TypeError("member profile must be a mapping")
        allowed = {
            "bandwidth", "capacity", "load", "failure_domain", "locality"
        }
        if set(profile) - allowed:
            raise ValueError("unknown member profile field")
        bandwidth = _positive_float(profile.get("bandwidth", 1.0), "bandwidth")
        capacity = _positive_float(profile.get("capacity", 1.0), "capacity")
        load = _nonnegative_float(profile.get("load", 0.0), "load")
        failure_domain = profile.get("failure_domain", member)
        locality = profile.get("locality", "")
        if not isinstance(failure_domain, str) or not isinstance(locality, str):
            raise TypeError("profile topology fields must be strings")
        result.append(
            (
                member,
                bandwidth,
                capacity,
                load,
                failure_domain,
                locality,
            )
        )
    return tuple(result)


def order_owners_for_client(
    owners: Iterable[str],
    member_profiles: Mapping[str, Mapping[str, Any]],
    client_locality: str,
) -> Tuple[str, ...]:
    values = tuple(owners)
    if not client_locality:
        return values
    return tuple(
        sorted(
            values,
            key=lambda member: (
                member_profiles.get(member, {}).get("locality")
                == client_locality
            ),
            reverse=True,
        )
    )


def _weighted_score(
    version: str,
    shard_index: int,
    member: str,
    profile: Tuple[float, float, float, str, str],
) -> float:
    digest = blake3.blake3(
        "{}\0{}\0{}".format(version, shard_index, member).encode("utf-8")
    ).digest()
    uniform = (int.from_bytes(digest, "big") + 1) / float(2**256 + 1)
    bandwidth, capacity, load, _, _ = profile
    weight = bandwidth * capacity / (1.0 + load)
    return weight / -math.log(uniform)


def _positive_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("{} must be positive".format(label))
    return result


def _nonnegative_float(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("{} must be non-negative".format(label))
    return result


def _checksum(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _integer(value: Any, label: str, minimum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError("invalid {}".format(label))
    return value
