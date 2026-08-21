#!/usr/bin/env python3
"""Fetch the audited Qwen checkpoint metadata without downloading weights.

The safetensors header is stored at the beginning of every shard. This script
uses two bounded HTTP Range requests per shard: eight bytes for the header
length, then exactly that many header bytes. It refuses a non-partial response
so a server change cannot silently turn this metadata operation into a weight
download.

The generated manifest is a weight-free CI fixture. Production loading never
uses it as a substitute for :meth:`CheckpointInventory.from_checkpoint`; A12
still validates the headers of the actual local shards before loading weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import httpx
from huggingface_hub import hf_hub_url

REPO_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAX_HEADER_BYTES = 4 * 1024 * 1024


def _range_get(client: httpx.Client, url: str, start: int, end: int) -> bytes:
    response = client.get(url, headers={"Range": f"bytes={start}-{end}"})
    if response.status_code != 206:
        raise RuntimeError(
            f"refusing non-partial response for {url}: HTTP {response.status_code}"
        )
    expected = end - start + 1
    if len(response.content) != expected:
        raise RuntimeError(
            f"range {start}-{end} returned {len(response.content)} bytes, expected {expected}"
        )
    content_range = response.headers.get("content-range", "")
    if not content_range.startswith(f"bytes {start}-{end}/"):
        raise RuntimeError(f"unexpected Content-Range for {url}: {content_range!r}")
    return response.content


def _fetch_header(client: httpx.Client, shard: str) -> tuple[dict[str, Any], str]:
    url = hf_hub_url(REPO_ID, shard, revision=REVISION)
    raw_length = _range_get(client, url, 0, 7)
    (length,) = struct.unpack("<Q", raw_length)
    if not 0 < length <= MAX_HEADER_BYTES:
        raise RuntimeError(f"unsafe safetensors header length for {shard}: {length}")
    raw_header = _range_get(client, url, 8, 8 + length - 1)
    return json.loads(raw_header.decode("utf-8")), hashlib.sha256(raw_header).hexdigest()


def build_manifest(index_path: Path) -> dict[str, Any]:
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    weight_map: dict[str, str] = index["weight_map"]
    shards = sorted(set(weight_map.values()))
    records: list[dict[str, Any]] = []
    header_hashes: dict[str, str] = {}

    with httpx.Client(follow_redirects=True, timeout=120) as client:
        for position, shard in enumerate(shards, start=1):
            header, header_hash = _fetch_header(client, shard)
            header_hashes[shard] = header_hash
            for name, metadata in header.items():
                if name == "__metadata__":
                    continue
                if weight_map.get(name) != shard:
                    raise RuntimeError(
                        f"header/index mismatch for {name}: {shard} vs {weight_map.get(name)}"
                    )
                records.append(
                    {
                        "name": name,
                        "shape": [int(value) for value in metadata["shape"]],
                        "dtype": metadata["dtype"],
                        "shard": shard,
                    }
                )
            print(f"[metadata] {position}/{len(shards)} {shard}", flush=True)

    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise RuntimeError("duplicate tensor name across safetensors headers")
    missing = sorted(set(weight_map) - set(names))
    unexpected = sorted(set(names) - set(weight_map))
    if missing or unexpected:
        raise RuntimeError(
            f"header/index tensor mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    return {
        "schema_version": 1,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "source": "HTTP Range reads of safetensors headers only; no weight payload bytes",
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "index_total_size": index.get("metadata", {}).get("total_size"),
        "header_sha256": header_hashes,
        "records": sorted(records, key=lambda record: record["name"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[metadata] wrote {args.output} ({len(manifest['records'])} tensors)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
