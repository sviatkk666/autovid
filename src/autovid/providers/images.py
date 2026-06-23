"""Unified image-search interface — find a ready image on the internet per scene.

This is the "search & download" approach (the generation approach is a separate
provider added later). Every source implements `search(query, orientation, limit)`
and returns `ImageResult`s carrying the download URL plus license/attribution so
the pipeline can credit images legally. Use `get_image_source(cfg)` to build one;
provider "auto" prefers no-key, license-aware sources (Openverse, Wikimedia),
then keyed stock (Pexels/Pixabay), then the Internet Archive.

Sources:
  - openverse        Creative Commons aggregator. No key. License metadata. (default)
  - wikimedia        Wikimedia Commons. No key. Mostly PD/CC.
  - pexels           Stock photos. Needs PEXELS_API_KEY.
  - pixabay          Stock photos. Needs PIXABAY_API_KEY.
  - internet_archive archive.org media. No key. Mixed licensing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

import requests

from ..config import env

USER_AGENT = "autovid/0.1 (https://example.com; image pipeline)"
_TIMEOUT = 20


@dataclass
class ImageResult:
    url: str                  # direct download URL
    source: str
    license: str = ""         # short name, e.g. "CC BY 2.0"
    license_url: str = ""
    attribution: str = ""     # creator / photographer
    title: str = ""
    credit_url: str = ""      # landing page to link back to
    width: int = 0
    height: int = 0


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    return s


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


class ImageSource(Protocol):
    name: str

    def search(self, query: str, orientation: str = "landscape", limit: int = 10) -> list[ImageResult]: ...


class OpenverseSource:
    name = "openverse"
    _ASPECT = {"landscape": "wide", "portrait": "tall"}

    def __init__(self, cfg: dict):
        self.http = _session()

    def search(self, query, orientation="landscape", limit=10):
        params = {
            "q": query,
            "page_size": limit,
            # only images we can reuse and modify commercially
            "license_type": "commercial,modification",
            "aspect_ratio": self._ASPECT.get(orientation, ""),
        }
        r = self.http.get("https://api.openverse.org/v1/images/", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        out = []
        for it in r.json().get("results", []):
            url = it.get("url")
            if not url:
                continue
            out.append(ImageResult(
                url=url, source=self.name,
                license=" ".join(x for x in [it.get("license", "").upper(), it.get("license_version", "")] if x).strip(),
                license_url=it.get("license_url", ""),
                attribution=it.get("creator", ""),
                title=it.get("title", ""),
                credit_url=it.get("foreign_landing_url", ""),
                width=it.get("width", 0) or 0,
                height=it.get("height", 0) or 0,
            ))
        return out


class WikimediaSource:
    name = "wikimedia"

    def __init__(self, cfg: dict):
        self.http = _session()

    def search(self, query, orientation="landscape", limit=10):
        params = {
            "action": "query", "format": "json", "generator": "search",
            "gsrsearch": query, "gsrnamespace": 6, "gsrlimit": limit,
            "prop": "imageinfo", "iiprop": "url|size|extmetadata",
        }
        r = self.http.get("https://commons.wikimedia.org/w/api.php", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        pages = (r.json().get("query") or {}).get("pages") or {}
        out = []
        for page in pages.values():
            info = (page.get("imageinfo") or [{}])[0]
            url = info.get("url")
            if not url:
                continue
            meta = info.get("extmetadata") or {}
            out.append(ImageResult(
                url=url, source=self.name,
                license=_strip_html((meta.get("LicenseShortName") or {}).get("value", "")),
                license_url=(meta.get("LicenseUrl") or {}).get("value", ""),
                attribution=_strip_html((meta.get("Artist") or {}).get("value", "")),
                title=page.get("title", ""),
                credit_url=info.get("descriptionurl", ""),
                width=info.get("width", 0) or 0,
                height=info.get("height", 0) or 0,
            ))
        return out


class PexelsSource:
    name = "pexels"
    _SIZE = {"landscape": "landscape", "portrait": "portrait"}

    def __init__(self, cfg: dict):
        key = env("PEXELS_API_KEY")
        if not key:
            raise RuntimeError("Pexels needs PEXELS_API_KEY in .env (free at pexels.com/api).")
        self.http = _session()
        self.http.headers["Authorization"] = key

    def search(self, query, orientation="landscape", limit=10):
        params = {"query": query, "per_page": limit, "orientation": self._SIZE.get(orientation, "landscape")}
        r = self.http.get("https://api.pexels.com/v1/search", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        out = []
        for p in r.json().get("photos", []):
            src = p.get("src") or {}
            url = src.get("large2x") or src.get("large") or src.get("original")
            if not url:
                continue
            out.append(ImageResult(
                url=url, source=self.name,
                license="Pexels License", license_url="https://www.pexels.com/license/",
                attribution=p.get("photographer", ""),
                title=p.get("alt", ""), credit_url=p.get("url", ""),
                width=p.get("width", 0) or 0, height=p.get("height", 0) or 0,
            ))
        return out


class PixabaySource:
    name = "pixabay"
    _ORIENT = {"landscape": "horizontal", "portrait": "vertical"}

    def __init__(self, cfg: dict):
        self.key = env("PIXABAY_API_KEY")
        if not self.key:
            raise RuntimeError("Pixabay needs PIXABAY_API_KEY in .env (free at pixabay.com/api/docs).")
        self.http = _session()

    def search(self, query, orientation="landscape", limit=10):
        params = {
            "key": self.key, "q": query, "per_page": max(3, limit),
            "image_type": "photo", "orientation": self._ORIENT.get(orientation, "all"),
        }
        r = self.http.get("https://pixabay.com/api/", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        out = []
        for h in r.json().get("hits", []):
            url = h.get("largeImageURL") or h.get("webformatURL")
            if not url:
                continue
            out.append(ImageResult(
                url=url, source=self.name,
                license="Pixabay License", license_url="https://pixabay.com/service/license/",
                attribution=h.get("user", ""), title=h.get("tags", ""),
                credit_url=h.get("pageURL", ""),
                width=h.get("imageWidth", 0) or 0, height=h.get("imageHeight", 0) or 0,
            ))
        return out


class InternetArchiveSource:
    name = "internet_archive"
    _IMG_FORMATS = ("JPEG", "PNG", "JPEG 2000", "Animated GIF")

    def __init__(self, cfg: dict):
        self.http = _session()

    def search(self, query, orientation="landscape", limit=10):
        params = {
            "q": f'({query}) AND mediatype:image', "rows": limit, "output": "json",
            "fl[]": "identifier",
        }
        r = self.http.get("https://archive.org/advancedsearch.php", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        docs = ((r.json().get("response") or {}).get("docs")) or []
        out = []
        for doc in docs:
            ident = doc.get("identifier")
            if not ident:
                continue
            result = self._first_image(ident)
            if result:
                out.append(result)
        return out

    def _first_image(self, ident: str) -> ImageResult | None:
        try:
            r = self.http.get(f"https://archive.org/metadata/{ident}", timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
        except Exception:
            return None
        meta = data.get("metadata") or {}
        for f in data.get("files") or []:
            if f.get("format") in self._IMG_FORMATS and f.get("name"):
                return ImageResult(
                    url=f"https://archive.org/download/{ident}/{f['name']}",
                    source=self.name,
                    license=meta.get("licenseurl", "") and "see license_url" or "unknown",
                    license_url=meta.get("licenseurl", ""),
                    attribution=meta.get("creator", "") if isinstance(meta.get("creator"), str) else "",
                    title=meta.get("title", "") if isinstance(meta.get("title"), str) else "",
                    credit_url=f"https://archive.org/details/{ident}",
                    width=int(f.get("width", 0) or 0), height=int(f.get("height", 0) or 0),
                )
        return None


_SOURCES = {
    "openverse": OpenverseSource,
    "wikimedia": WikimediaSource,
    "pexels": PexelsSource,
    "pixabay": PixabaySource,
    "internet_archive": InternetArchiveSource,
}


def get_image_source(cfg: dict) -> ImageSource:
    icfg = cfg.get("images", {})
    provider = icfg.get("provider", "auto")

    if provider == "auto":
        if env("PEXELS_API_KEY"):
            provider = "pexels"
        elif env("PIXABAY_API_KEY"):
            provider = "pixabay"
        else:
            provider = "openverse"  # no key, license-aware — safe default

    cls = _SOURCES.get(provider)
    if cls is None:
        raise ValueError(f"Unknown images.provider: {provider}")
    return cls(icfg)
