"""GLM media affinity and per-media encoder dispatch plans."""

import hashlib
import json
import logging
import os
from typing import Optional, Tuple

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import Modality

logger = logging.getLogger(__name__)
_ENCODER_MEDIA_HASH_HEX_LENGTH = 16
_encoder_affinity_invalid_shards_warned = False


def create_part_req_id(original_req_id: str, part_idx: int) -> str:
    """Create a unique part request ID by appending part index suffix."""
    return f"{original_req_id}_local_part_{part_idx}"


def _encoder_affinity_shard_id(req_id: Optional[str]) -> Tuple[int, int]:
    global _encoder_affinity_invalid_shards_warned

    affinity_shards = envs.GLM_ENCODER_AFFINITY_SHARDS.get()
    if affinity_shards < 1:
        if not _encoder_affinity_invalid_shards_warned:
            logger.warning(
                "Invalid GLM_ENCODER_AFFINITY_SHARDS=%s; falling back to 1",
                affinity_shards,
            )
            _encoder_affinity_invalid_shards_warned = True
        affinity_shards = 1
    if affinity_shards == 1:
        return 0, affinity_shards
    if not req_id:
        raise ValueError("req_id is required when encoder affinity sharding is enabled")

    request_hash = hashlib.sha256(req_id.encode("utf-8")).digest()
    shard_id = int.from_bytes(request_hash[:8], byteorder="big") % affinity_shards
    return shard_id, affinity_shards


def create_encoder_session_id(
    upstream_session_id: Optional[str],
    media_identifier,
    req_id: Optional[str] = None,
) -> str:
    if isinstance(media_identifier, bytes):
        media_bytes = media_identifier
    elif isinstance(media_identifier, str):
        media_bytes = media_identifier.encode("utf-8")
    else:
        media_bytes = json.dumps(
            media_identifier,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")

    media_hash = hashlib.sha256(media_bytes).hexdigest()[
        :_ENCODER_MEDIA_HASH_HEX_LENGTH
    ]
    session_id = (
        f"{upstream_session_id}_{media_hash}" if upstream_session_id else media_hash
    )

    # Keep K=1 byte-for-byte compatible with the original media affinity. For
    # K>1, distribute a hot media key over a bounded number of LB affinity
    # keys while keeping /encode and /send for the same part on one encoder.
    shard_id, affinity_shards = _encoder_affinity_shard_id(req_id)
    if affinity_shards == 1:
        return session_id

    # Re-hash the bounded affinity key so the LB always receives the original
    # fixed-width 16-character hexadecimal format. This remains compatible
    # with LBs that parse Session-Id as a uint64 or validate it as hex.
    sharded_key = f"{session_id}:{shard_id}".encode("utf-8")
    return hashlib.sha256(sharded_key).hexdigest()[:_ENCODER_MEDIA_HASH_HEX_LENGTH]


def _encoder_request_headers(req_id: str, encoder_session_id: Optional[str] = None):
    headers = {"Request-Id": req_id}
    if encoder_session_id and envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.get():
        headers["Session-Id"] = encoder_session_id
    return headers


def _video_max_frames(video_item) -> Optional[int]:
    """Return a request-provided sampled-frame cap, if present."""
    url = video_item.get("url")
    if isinstance(url, dict):
        max_frames = url.get("max_frames")
        if max_frames is not None:
            try:
                return int(max_frames)
            except (TypeError, ValueError):
                return None
    return None


def _video_is_framed(url) -> bool:
    """Whether the source is already a list of decoded video frames."""
    if isinstance(url, dict):
        url = url.get("url")
    return isinstance(url, (list, tuple))


def _video_size_bytes(url) -> Optional[int]:
    """Return a local/in-memory video's size without doing network IO."""
    if isinstance(url, dict):
        url = url.get("url")
    try:
        if isinstance(url, (bytes, bytearray)):
            return len(url)
        if isinstance(url, str):
            if url.startswith("data:"):
                return len(url)
            path = url[len("file://") :] if url.startswith("file://") else url
            if os.path.isfile(path):
                return os.path.getsize(path)
    except Exception:
        pass
    return None


def _should_shard_video(video_item, num_encoders) -> bool:
    """Whether one video is large enough to split across all encoders."""
    url = video_item.get("url")
    if _video_is_framed(url):
        return False

    min_frames_per_encoder = (
        envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_FRAMES_PER_ENCODER.get()
    )
    max_frames = _video_max_frames(video_item)
    if max_frames is not None and max_frames < num_encoders * min_frames_per_encoder:
        return False

    min_bytes = envs.SGLANG_ENCODER_VIDEO_SHARD_MIN_MB.get() * 1024 * 1024
    size_bytes = _video_size_bytes(url)
    if size_bytes is not None and size_bytes < min_bytes:
        return False
    return True


def build_glm_encode_requests(
    req_id,
    mm_data,
    assignments,
    urls,
    host,
    time_stats_json=None,
    upstream_session_id=None,
):
    requests = []
    videos = [item for item in mm_data if item["modality"] == Modality.VIDEO]
    shard_video = (
        len(videos) == 1 and len(urls) > 1 and _should_shard_video(videos[0], len(urls))
    )
    for modality, counts in assignments.items():
        items = [item for item in mm_data if item["modality"] == modality]
        if modality == Modality.VIDEO and shard_video:
            routed = [(idx, items[0], idx) for idx in range(len(urls))]
        else:
            routed = []
            offset = 0
            for idx, count in enumerate(counts):
                routed.extend(
                    (idx, item, None) for item in items[offset : offset + count]
                )
                offset += count
        for idx, item, shard in routed:
            part_idx = len(requests)
            part_rid = create_part_req_id(req_id, part_idx)
            # Preserve main's validated media hash and preprocessing options.
            media = {key: value for key, value in item.items() if key != "modality"}
            if isinstance(media.get("url"), dict):
                inner = media.pop("url")
                media = {**inner, **media}
            payload = dict(
                encoder_idx=idx,
                encoder_url=urls[idx],
                mm_items=[media],
                part_idx=part_idx,
                req_id=part_rid,
                modality=modality.name,
                prefill_host=host,
                embedding_port=None,
                time_stats_json=time_stats_json,
            )
            if shard is not None:
                payload["video_num_shards"] = len(urls)
                payload["video_shard_idx"] = shard
                media["_glm_video_shard"] = [shard, len(urls)]
            if envs.GLM_ENABLE_ENCODER_SESSION_ID_HEADER.get():
                payload["encoder_session_id"] = create_encoder_session_id(
                    upstream_session_id, media, req_id=part_rid
                )
            requests.append(payload)
    for payload in requests:
        payload["num_parts"] = len(requests)
    return requests


def encoder_part_routes(requests):
    # Media payloads stay in the dispatch thread; the scheduler needs only routing.
    keys = (
        "req_id",
        "part_idx",
        "num_parts",
        "modality",
        "encoder_idx",
        "encoder_session_id",
    )
    return [{key: req[key] for key in keys if key in req} for req in requests]
