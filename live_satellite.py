from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from io import BytesIO
from typing import Any

import numpy as np
import tifffile


CDSE_STAC_SEARCH_URL = os.getenv(
    "LIVE_SATELLITE_STAC_URL",
    "https://stac.dataspace.copernicus.eu/v1/search",
).strip()
CDSE_TOKEN_URL = os.getenv(
    "LIVE_SATELLITE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
).strip()
CDSE_PROCESS_URL = os.getenv(
    "LIVE_SATELLITE_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/process",
).strip()


def live_satellite_enabled() -> bool:
    raw = os.getenv("ENABLE_LIVE_SATELLITE_API", "1").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def credentials_configured() -> bool:
    return bool(
        os.getenv("LIVE_SATELLITE_CLIENT_ID", "").strip()
        and os.getenv("LIVE_SATELLITE_CLIENT_SECRET", "").strip()
    )


def _post_json(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected JSON payload from {url}")
    return data


def _post_form(url: str, form_data: dict[str, Any]) -> dict[str, Any]:
    encoded = urllib.parse.urlencode(form_data).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.load(response)
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected OAuth payload from {url}")
    return data


def _post_bytes(url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> bytes:
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=req_headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as response:
        return response.read()


def _to_channels_last(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr[..., None]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image dimensions: {arr.shape}")
    if arr.shape[0] <= 16 and arr.shape[1] == arr.shape[2]:
        return np.moveaxis(arr, 0, -1)
    return arr


def _parse_utc_text(value: str | None) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)
    except Exception:
        return None


def _sensor_collection(sensor: str) -> str:
    sensor_key = str(sensor or "").strip().upper()
    if sensor_key == "S1":
        return "sentinel-1-grd"
    if sensor_key == "S2":
        return "sentinel-2-l2a"
    raise ValueError(f"Unsupported live sensor: {sensor}")


def _scene_summary(feature: dict[str, Any]) -> dict[str, Any]:
    props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
    bbox = feature.get("bbox")
    return {
        "id": feature.get("id"),
        "collection": feature.get("collection"),
        "datetime_utc": props.get("datetime"),
        "bbox": bbox if isinstance(bbox, list) and len(bbox) == 4 else None,
        "platform": props.get("platform"),
        "instruments": props.get("instruments"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "polarizations": props.get("sar:polarizations"),
        "orbit_state": props.get("sat:orbit_state"),
        "assets_count": len(feature.get("assets", {}) or {}),
    }


def search_latest_scene(
    *,
    sensor: str,
    bbox: list[float],
    days_back: int = 14,
    limit: int = 5,
    date_from: str | None = None,
    date_to: str | None = None,
    max_cloud_cover: float = 35.0,
) -> dict[str, Any]:
    now_utc = datetime.now(UTC)
    dt_to = _parse_utc_text(date_to) or now_utc
    dt_from = _parse_utc_text(date_from) or (dt_to - timedelta(days=max(1, int(days_back))))
    payload = {
        "collections": [_sensor_collection(sensor)],
        "bbox": [float(x) for x in bbox],
        "limit": max(1, min(int(limit), 20)),
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "datetime": f"{dt_from.isoformat().replace('+00:00', 'Z')}/{dt_to.isoformat().replace('+00:00', 'Z')}",
    }
    data = _post_json(CDSE_STAC_SEARCH_URL, payload)
    features = data.get("features", [])
    if not isinstance(features, list) or not features:
        raise RuntimeError("No satellite scenes found for the requested sensor/bbox/time window.")
    sensor_key = str(sensor).strip().upper()
    selected: dict[str, Any] | None = None
    for feature in features:
        if not isinstance(feature, dict):
            continue
        if sensor_key == "S2":
            props = feature.get("properties", {}) if isinstance(feature.get("properties"), dict) else {}
            cloud = props.get("eo:cloud_cover")
            try:
                if cloud is not None and float(cloud) > float(max_cloud_cover):
                    continue
            except Exception:
                pass
        selected = feature
        break
    if selected is None:
        raise RuntimeError("Scenes were found, but all matching S2 scenes exceeded the cloud-cover limit.")
    return _scene_summary(selected)


def _oauth_token() -> str:
    client_id = os.getenv("LIVE_SATELLITE_CLIENT_ID", "").strip()
    client_secret = os.getenv("LIVE_SATELLITE_CLIENT_SECRET", "").strip()
    if not client_id or not client_secret:
        raise RuntimeError(
            "Live satellite credentials are missing. Set LIVE_SATELLITE_CLIENT_ID and LIVE_SATELLITE_CLIENT_SECRET."
        )
    payload = _post_form(
        CDSE_TOKEN_URL,
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
    )
    token = str(payload.get("access_token", "")).strip()
    if not token:
        raise RuntimeError("OAuth token response did not include access_token.")
    return token


def _s2_evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"],
      units: "REFLECTANCE"
    }],
    output: { bands: 9, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.B02, sample.B03, sample.B04, sample.B05, sample.B06, sample.B07, sample.B08, sample.B11, sample.B12];
}
""".strip()


def _s1_evalscript() -> str:
    return """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["VV", "VH"]
    }],
    output: { bands: 2, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(sample) {
  return [sample.VV, sample.VH];
}
""".strip()


def _build_process_payload(
    *,
    sensor: str,
    bbox: list[float],
    width: int,
    height: int,
    scene_datetime_utc: str | None,
    max_cloud_cover: float,
) -> dict[str, Any]:
    scene_dt = _parse_utc_text(scene_datetime_utc)
    if scene_dt is None:
        scene_dt = datetime.now(UTC)
    time_from = (scene_dt - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    time_to = (scene_dt + timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    sensor_key = str(sensor).strip().upper()
    data_item: dict[str, Any] = {
        "type": _sensor_collection(sensor_key),
        "dataFilter": {
            "timeRange": {"from": time_from, "to": time_to},
            "mosaickingOrder": "mostRecent",
        },
    }
    evalscript = _s2_evalscript()
    if sensor_key == "S2":
        data_item["dataFilter"]["maxCloudCoverage"] = float(max_cloud_cover)
    else:
        evalscript = _s1_evalscript()
        data_item["processing"] = {
            "orthorectify": True,
            "backCoeff": "SIGMA0_ELLIPSOID",
        }
    return {
        "input": {
            "bounds": {
                "bbox": [float(x) for x in bbox],
                "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"},
            },
            "data": [data_item],
        },
        "output": {
            "width": int(width),
            "height": int(height),
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": evalscript,
    }


def build_synthetic_geo_meta(
    *,
    bbox: list[float],
    width: int,
    height: int,
    channels: int,
    scene_datetime_utc: str | None,
) -> dict[str, Any]:
    min_lon, min_lat, max_lon, max_lat = [float(x) for x in bbox]
    sx = (max_lon - min_lon) / max(1, int(width))
    sy = (max_lat - min_lat) / max(1, int(height))
    return {
        "status": "ok",
        "path": None,
        "warnings": [],
        "epsg": 4326,
        "pixel_scale": [float(sx), float(sy)],
        "tiepoint": [float(min_lon), float(max_lat)],
        "center_lat": float((min_lat + max_lat) / 2.0),
        "center_lon": float((min_lon + max_lon) / 2.0),
        "image_datetime_utc": scene_datetime_utc,
        "has_model_tiepoint": True,
        "has_geotiff_metadata": True,
        "image_shape": [int(height), int(width), int(channels)],
        "georeference_source": "stac_bbox_process_request",
    }


def fetch_latest_chip(
    *,
    sensor: str,
    bbox: list[float],
    width: int = 256,
    height: int = 256,
    days_back: int = 14,
    date_from: str | None = None,
    date_to: str | None = None,
    max_cloud_cover: float = 35.0,
) -> dict[str, Any]:
    scene = search_latest_scene(
        sensor=sensor,
        bbox=bbox,
        days_back=days_back,
        date_from=date_from,
        date_to=date_to,
        max_cloud_cover=max_cloud_cover,
    )
    token = _oauth_token()
    payload = _build_process_payload(
        sensor=sensor,
        bbox=bbox,
        width=width,
        height=height,
        scene_datetime_utc=scene.get("datetime_utc"),
        max_cloud_cover=max_cloud_cover,
    )
    raw_tiff = _post_bytes(
        CDSE_PROCESS_URL,
        payload,
        headers={"Authorization": f"Bearer {token}"},
    )
    arr = np.asarray(tifffile.imread(BytesIO(raw_tiff)))
    arr = _to_channels_last(arr).astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    geo_meta = build_synthetic_geo_meta(
        bbox=bbox,
        width=int(arr.shape[1]),
        height=int(arr.shape[0]),
        channels=int(arr.shape[-1]),
        scene_datetime_utc=scene.get("datetime_utc"),
    )
    return {
        "scene": scene,
        "image": arr,
        "raw_tiff_bytes": raw_tiff,
        "geo_meta": geo_meta,
    }
