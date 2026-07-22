"""Fetch an openpi checkpoint from GCS over plain HTTPS, in parallel, resumably.

openpi's own downloader is slow here for two independent reasons: it inherits HTTP_PROXY
from the environment, and it fetches everything on a single thread. This bypasses both --
it talks straight to storage.googleapis.com with the proxy disabled and splits large
objects into byte ranges. The bucket honors accept-ranges, so partial chunks resume.

Writes into the openpi cache in the layout maybe_download() expects, so afterwards the
inference script finds the checkpoint already present and skips downloading entirely.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import urllib.parse
import urllib.request

BUCKET = "openpi-assets"
CHUNK = 32 * 1024 * 1024  # 32 MiB range requests
MAX_ATTEMPTS = 6

# Talk to GCS directly. The local proxy caps a single stream at ~0.065 MB/s and actually
# degrades under concurrency; bypassing it gets ~3.5 MB/s across 8 streams.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

_done_bytes = 0
_lock = threading.Lock()


def _advance(n: int) -> None:
    global _done_bytes
    with _lock:
        _done_bytes += n


def list_objects(prefix: str) -> list[tuple[str, int]]:
    base = (
        f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
        f"?prefix={urllib.parse.quote(prefix)}&fields=items(name,size),nextPageToken"
    )
    items: list[tuple[str, int]] = []
    token = None
    while True:
        with _OPENER.open(base + (f"&pageToken={token}" if token else ""), timeout=60) as r:
            body = json.load(r)
        items += [(i["name"], int(i["size"])) for i in body.get("items", [])]
        token = body.get("nextPageToken")
        if not token:
            break
    # Zero-byte objects ending in "/" are directory placeholders, not real files.
    return [(n, s) for n, s in items if not n.endswith("/")]


def fetch_range(name: str, start: int, end: int, dest: pathlib.Path) -> None:
    """Download [start, end) of an object into a sidecar chunk file, skipping completed chunks."""
    part = dest.parent / f".{dest.name}.part.{start}"
    want = end - start
    if part.exists() and part.stat().st_size == want:
        _advance(want)
        return

    url = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(name)}"
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})

    for attempt in range(MAX_ATTEMPTS):
        written = 0
        try:
            with _OPENER.open(req, timeout=120) as r, part.open("wb") as f:
                while buf := r.read(1 << 20):
                    f.write(buf)
                    written += len(buf)
                    _advance(len(buf))
            if part.stat().st_size != want:
                raise OSError(f"short chunk: {part.stat().st_size} != {want}")
            return
        except Exception as e:  # noqa: BLE001 - retry any transport-level failure
            _advance(-written)  # roll back this attempt's contribution to the progress meter
            part.unlink(missing_ok=True)
            if attempt == MAX_ATTEMPTS - 1:
                raise
            print(f"  retry {name.rsplit('/', 1)[-1]} @{start}: {e}", file=sys.stderr, flush=True)
            time.sleep(2**attempt)


def assemble(dest: pathlib.Path, size: int) -> None:
    """Concatenate the .part.* chunks into the final file, then drop the chunks."""
    parts = sorted(dest.parent.glob(f".{dest.name}.part.*"), key=lambda p: int(p.name.rsplit(".", 1)[-1]))
    with dest.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                shutil.copyfileobj(f, out, 1 << 22)
    if dest.stat().st_size != size:
        raise RuntimeError(f"{dest}: assembled {dest.stat().st_size} bytes, expected {size}")
    for p in parts:
        p.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/pi05_base", help="path within the openpi-assets bucket")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cache = pathlib.Path(os.getenv("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
    root = cache / BUCKET / args.checkpoint
    if root.exists():
        print(f"already present, nothing to do: {root}")
        return

    objects = list_objects(args.checkpoint + "/")
    total = sum(s for _, s in objects)
    print(f"{len(objects)} objects, {total / 1e9:.2f} GB -> {root}", flush=True)

    staging = root.with_suffix(".partial")
    jobs: list[tuple[str, int, int, pathlib.Path]] = []
    for name, size in objects:
        dest = staging / name[len(args.checkpoint) + 1 :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size == size:
            _advance(size)  # already assembled by an earlier run
            continue
        for start in range(0, max(size, 1), CHUNK):
            jobs.append((name, start, min(start + CHUNK, size), dest))

    started = time.time()
    stop = threading.Event()

    def report() -> None:
        while not stop.wait(5):
            got = _done_bytes
            rate = got / max(time.time() - started, 1e-9)
            eta = f"{(total - got) / rate / 60:.0f} min" if rate > 0 else "?"
            print(
                f"  {got / 1e9:5.2f}/{total / 1e9:.2f} GB ({100 * got / total:4.1f}%)  "
                f"{rate / 1e6:5.2f} MB/s  eta {eta}",
                flush=True,
            )

    threading.Thread(target=report, daemon=True).start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for f in concurrent.futures.as_completed([pool.submit(fetch_range, *j) for j in jobs]):
                f.result()
    finally:
        stop.set()

    print("assembling chunks...", flush=True)
    for name, size in objects:
        dest = staging / name[len(args.checkpoint) + 1 :]
        if not (dest.exists() and dest.stat().st_size == size):
            assemble(dest, size)

    staging.rename(root)
    print(f"done in {(time.time() - started) / 60:.1f} min -> {root}", flush=True)


if __name__ == "__main__":
    main()
