"""Network-aware online/offline provider routing with bounded timeouts.

Provider functions are injected by the application; this module does not
assume a particular cloud API or claim that internet reachability guarantees a
specific provider is healthy.
"""
from __future__ import annotations
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen

def internet_available(*, probe_urls: tuple[str, ...] = ("https://www.google.com/generate_204", "https://www.cloudflare.com/cdn-cgi/trace"), timeout: float = 2.0) -> bool:
    """Probe actual HTTP reachability with a short timeout (not Wi-Fi status).

    Returns False if all probes fail. A reachable probe does not guarantee that
    a particular AI/API provider is healthy; the provider call must still handle
    its own timeout, authentication, rate limits, and errors.
    """
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    for url in probe_urls:
        # urlopen also accepts file:// and ftp://; only plain web probes are allowed.
        if not str(url).lower().startswith(("http://", "https://")):
            raise ValueError(f"probe URL must use http or https: {url!r}")
    for url in probe_urls:
        try:
            request = Request(url, headers={"User-Agent": "minimemory-connectivity/0.8.2"}, method="GET")
            with urlopen(request, timeout=float(timeout)) as response:
                if 200 <= int(response.status) < 400:
                    return True
        except Exception:
            continue
    return False


class ResourceRouter:
    def __init__(self, *, online_available: Callable[[], bool], online_provider: Optional[Callable[..., Any]] = None,
                 local_provider: Optional[Callable[..., Any]] = None):
        self.online_available = online_available
        self.online_provider = online_provider
        self.local_provider = local_provider

    def run(self, *args: Any, prefer_online: bool = True, **kwargs: Any) -> dict:
        online = False
        try: online = bool(self.online_available())
        except Exception: online = False
        attempts=[]
        order = ([('online', self.online_provider), ('local', self.local_provider)] if prefer_online and online
                 else [('local', self.local_provider), ('online', self.online_provider)] if not prefer_online
                 else [('local', self.local_provider), ('online', self.online_provider)])
        # Offline-first when the network check fails; online is attempted only if available.
        for name, provider in order:
            if provider is None: continue
            if name == 'online' and not online: continue
            try:
                value = provider(*args, **kwargs)
                return {"ok": True, "mode": name, "result": value, "attempts": attempts}
            except Exception as exc:
                attempts.append({"mode": name, "error": type(exc).__name__})
        return {"ok": False, "mode": "unavailable", "result": None, "attempts": attempts,
                "message": "No configured provider completed the request."}
