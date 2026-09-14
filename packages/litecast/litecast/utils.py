"""Utility functions for the litecast package."""

import os
import blake3
import logging
import re
from pathlib import Path
from typing import List, Dict

from litecast.envs import LITECAST_VERSION_PREFIX, LITECAST_LOG_LEVEL

# Configure logging for this module only
logger = logging.getLogger(__name__)
logger.setLevel(LITECAST_LOG_LEVEL)

# Only add handler if logger doesn't already have handlers
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setLevel(LITECAST_LOG_LEVEL)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def compute_checksum(file_path: str) -> str:
    """Compute the blake3 checksum of a file using memory mapping for optimal performance.

    Args:
        file_path: Path to the file

    Returns:
        blake3 checksum as a hex string
    """
    hasher = blake3.blake3(max_threads=blake3.blake3.AUTO)
    hasher.update_mmap(file_path)
    return hasher.hexdigest()


def verify_checksum(file_path: str, expected_checksum: str) -> bool:
    """Verify the checksum of a file.

    Args:
        file_path: Path to the file
        expected_checksum: Expected blake3 checksum

    Returns:
        True if the checksum matches, False otherwise
    """
    actual_checksum = compute_checksum(file_path)
    return actual_checksum == expected_checksum


def get_next_version(base_dir: str) -> str:
    """Get the next version folder name.

    Args:
        base_dir: Base directory for version folders

    Returns:
        Next version folder name (e.g., 'v1', 'v2', etc.)
    """
    version_pattern = re.compile(f"^{LITECAST_VERSION_PREFIX}(\\d+)$")
    version = 0

    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)):
                match = version_pattern.match(item)
                if match:
                    current_version = int(match.group(1))
                    version = max(version, current_version)

    return f"{LITECAST_VERSION_PREFIX}{version + 1}"


def get_all_versions(base_dir: str) -> List[str]:
    """Get all version folders sorted by version number.

    Args:
        base_dir: Base directory for version folders

    Returns:
        List of version folder names sorted by version number
    """
    version_pattern = re.compile(f"^{LITECAST_VERSION_PREFIX}(\\d+)$")
    versions = []

    if os.path.exists(base_dir):
        for item in os.listdir(base_dir):
            if os.path.isdir(os.path.join(base_dir, item)):
                match = version_pattern.match(item)
                if match:
                    versions.append((int(match.group(1)), item))

    # Sort by version number
    versions.sort()
    return [v[1] for v in versions]


def parse_distribution_file(file_path: str) -> Dict[str, str]:
    """Parse the distribution file.

    Args:
        file_path: Path to the distribution file

    Returns:
        Dictionary mapping version folder names to info string (checksum|num_shards or just checksum for backward compatibility)
    """
    result = {}
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and ":" in line:
                    version, info = line.split(":", 1)
                    result[version.strip()] = info.strip()
    return result


def update_distribution_file(file_path: str, version: str, checksum: str, max_versions: int, num_shards: int) -> List[str]:
    """Update the distribution file with a new version.

    Args:
        file_path: Path to the distribution file
        version: Version folder name
        checksum: Checksum of the original file
        max_versions: Maximum number of versions to keep
        num_shards: Number of shards for this version

    Returns:
        List of versions to remove (if any)
    """
    versions_to_remove = []

    # Read existing distribution file
    distribution = parse_distribution_file(file_path)

    # Add or update the new version with checksum and number of shards
    distribution[version] = f"{checksum}|{num_shards}"

    # Remove oldest versions if exceeding max_versions
    if len(distribution) > max_versions:
        # Get versions sorted by number
        sorted_versions = sorted(
            distribution.keys(),
            key=lambda v: int(v[len(LITECAST_VERSION_PREFIX) :]) if v.startswith(LITECAST_VERSION_PREFIX) else 0,
        )

        # Determine versions to remove
        versions_to_remove = sorted_versions[: len(distribution) - max_versions]

        # Remove from distribution dictionary
        for v in versions_to_remove:
            del distribution[v]

    # Write updated distribution file
    with open(file_path, "w") as f:
        for v, info in sorted(
            distribution.items(),
            key=lambda item: int(item[0][len(LITECAST_VERSION_PREFIX) :]) if item[0].startswith(LITECAST_VERSION_PREFIX) else 0,
        ):
            f.write(f"{v}: {info}\n")

    return versions_to_remove


def remove_distribution_versions(file_path: str, versions: List[str]) -> None:
    """Remove versions from a manifest and atomically rewrite the remainder."""
    remove = set(versions)
    if not remove:
        return
    distribution = parse_distribution_file(file_path)
    for version in remove:
        distribution.pop(version, None)
    temporary = file_path + ".tmp"
    with open(temporary, "w") as handle:
        for version, info in sorted(
            distribution.items(),
            key=lambda item: int(item[0][len(LITECAST_VERSION_PREFIX) :])
            if item[0].startswith(LITECAST_VERSION_PREFIX)
            else 0,
        ):
            handle.write("{}: {}\n".format(version, info))
    os.replace(temporary, file_path)


def get_shard_count(file_size: int, shard_size: int) -> int:
    """Calculate the number of shards for a file.

    Args:
        file_size: Size of the file in bytes
        shard_size: Size of each shard in bytes

    Returns:
        Number of shards required
    """
    return (file_size + shard_size - 1) // shard_size


def get_shard_filename(shard_index: int) -> str:
    """Get the filename for a shard.

    Args:
        shard_index: Index of the shard (0-based)

    Returns:
        Shard filename (e.g., 'shard_001.bin')
    """
    return f"shard_{shard_index + 1:05d}.bin"


def ensure_dir(directory: str) -> None:
    """Ensure a directory exists.

    Args:
        directory: Directory path
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
