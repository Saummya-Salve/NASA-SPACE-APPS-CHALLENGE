"""
pipeline.py - turns SPHEREx cutouts into aligned, stacked frames for the website.

What it does for a target + detector band:
  1. ensure_data(): downloads any missing cutouts live from IRSA (cached 24 h)
  2. puts every cutout onto ONE north-up pixel grid fixed on the sky
     (so background stars stay still and movers move)
  3. groups images into epochs (one per six-month sky pass) and stacks each epoch
  4. saves PNG frames, a blink GIF (first vs last), a difference image
     (last minus first) and a deep image (all epochs together)
  5. writes manifest.json in the API-contract format, under static/frames/...

Usage (project folder, .venv active):
    python pipeline.py --target barnard --band D1
    python pipeline.py --target epsindib --band D6
    python pipeline.py --ra 269.4463 --dec 4.7677 --band D2
Then open static/frames/<name>/ to look at the PNGs.

Needs: pip install reproject pillow   (plus what fetch_spherex.py needs)
"""

import argparse
import json
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import astropy.units as u
import matplotlib
import numpy as np
from astropy import log as astropy_log
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning
from PIL import Image
from reproject import reproject_interp

from fetch_spherex import ALIASES, PRESETS, download, move_to_now, query_spherex

warnings.simplefilter("ignore", FITSFixedWarning)
astropy_log.setLevel("WARNING")

DEFAULT_SIZE = 0.05        # cutout size in degrees (same default as fetch_spherex.py)
CACHE_HOURS = 24           # re-query IRSA after this long, so new releases appear
IRSA_DOWN_MINUTES = 15     # after IRSA fails, don't try again for this long (keeps the site fast)
_IRSA_DOWN_MARK = Path("data") / ".irsa_down"
GRID_SCALE = 3.0           # arcsec per output pixel (SPHEREx pixels are ~6.2")
UPSCALE = 4                # PNGs are enlarged 4x (nearest neighbour) for crisp display
STATIC = Path("static")
FRAMES_DIR = STATIC / "frames"


# ---------------------------------------------------------------------------
# Target and data
# ---------------------------------------------------------------------------
def resolve(target=None, ra=None, dec=None):
    """Return (key or None, folder label, ra, dec) for a preset name or raw coordinates."""
    if target:
        key = ALIASES.get(re.sub(r"[^a-z0-9]", "", target.lower()))
        if key is None:
            raise ValueError(f"Unknown target '{target}'. Known: {', '.join(PRESETS)}")
        p = PRESETS[key]
        c = move_to_now(p["ra"], p["dec"], p["pmra"], p["pmdec"], p["ref"])
        return key, p["label"], float(c.ra.deg), float(c.dec.deg)
    if ra is None or dec is None:
        raise ValueError("Give a target name or both ra and dec.")
    return None, f"ra{ra:.4f}_dec{dec:+.4f}", float(ra), float(dec)


def data_folder(label, size, data_dir="data"):
    same = abs(size - DEFAULT_SIZE) < 1e-9
    return Path(data_dir) / (label if same else f"{label}_size{size:g}")


def _missing_msg(folder, bands, label):
    have = [b for b in ("D1", "D2", "D3", "D4", "D5", "D6") if any(folder.glob(f"*_{b}_*.fits"))]
    want = ", ".join(b.upper() for b in bands)
    if have:
        return (f"Detector {want} is not downloaded for {label.replace('_', ' ')}, and NASA's IRSA archive is "
                f"not responding right now. Available without IRSA: {', '.join(have)}.")
    return ("NASA's IRSA archive is not responding right now and there is no saved data for this "
            "position yet. Please try again in a few minutes.")


def ensure_data(target=None, ra=None, dec=None, bands=("D1",), size=DEFAULT_SIZE,
                max_per_band=40, data_dir="data", workers=3):
    """
    Make sure cutouts exist locally; download missing ones live from IRSA.
    The IRSA search result is cached for CACHE_HOURS, so repeated requests are fast
    but new data releases still show up within a day.
    Returns (folder, latest_image_date or None).
    """
    key, label, ra0, dec0 = resolve(target, ra, dec)
    folder = data_folder(label, size, data_dir)
    folder.mkdir(parents=True, exist_ok=True)

    qfile = folder / f"irsa_query_{size:g}.ecsv"
    irsa_recently_down = (_IRSA_DOWN_MARK.exists() and
                          time.time() - _IRSA_DOWN_MARK.stat().st_mtime < IRSA_DOWN_MINUTES * 60)
    have_all = all(any(folder.glob(f"*_{b.upper()}_*.fits")) for b in bands)
    if qfile.exists() and time.time() - qfile.stat().st_mtime < CACHE_HOURS * 3600:
        table = Table.read(qfile)
    elif irsa_recently_down and (have_all or qfile.exists()):
        print("[pipeline] IRSA was down a few minutes ago; using saved data without asking again")
        if qfile.exists():
            table = Table.read(qfile)
        else:
            return folder, None
    elif irsa_recently_down:
        raise RuntimeError(_missing_msg(folder, bands, label))
    else:
        print(f"[pipeline] querying IRSA for {label} (1-3 min)...")
        try:
            table = query_spherex(SkyCoord(ra0 * u.deg, dec0 * u.deg), size)
            table.write(qfile, overwrite=True)
        except Exception as exc:
            # IRSA busy or down: fall back to what we already have instead of failing
            print(f"[pipeline] IRSA not responding ({type(exc).__name__}: {str(exc)[:80]})")
            _IRSA_DOWN_MARK.parent.mkdir(parents=True, exist_ok=True)
            _IRSA_DOWN_MARK.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
            if qfile.exists():
                print("[pipeline] using the previous IRSA search result")
                table = Table.read(qfile)
            elif have_all:
                print("[pipeline] using the images already downloaded")
                return folder, None
            else:
                raise RuntimeError(_missing_msg(folder, bands, label)) from exc
    if len(table) == 0:
        return folder, None

    latest = Time(float(np.max(np.asarray(table["mjd"], dtype=float))), format="mjd").iso[:10]

    jobs = []
    for band in bands:
        band = band.upper()
        rows = table[np.array([str(b) == f"SPHEREx-{band}" for b in table["band"]])]
        if len(rows) == 0:
            continue
        # Already have (almost) enough images for this band? Then don't download more.
        have = len(list(folder.glob(f"*_{band}_*.fits")))
        if have >= 0.75 * min(len(rows), max_per_band):
            continue
        if len(rows) > max_per_band:  # same even-in-time selection as fetch_spherex.py
            idx = np.unique(np.linspace(0, len(rows) - 1, max_per_band).round().astype(int))
            rows = rows[idx]
        dates = Time(np.asarray(rows["mjd"], dtype=float), format="mjd")
        for i, (row, d) in enumerate(zip(rows, dates), 1):
            path = folder / f"{d.iso[:10]}_{band}_{i:03d}.fits"
            if not (path.exists() and path.stat().st_size > 0):
                jobs.append((str(row["cutout_url"]), path))

    if jobs:
        print(f"[pipeline] downloading {len(jobs)} missing cutouts...")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda j: download(*j), jobs))
    return folder, latest


# ---------------------------------------------------------------------------
# Loading, aligning and stacking
# ---------------------------------------------------------------------------
def obs_time(path, headers):
    for h in headers:
        for k in ("MJD-AVG", "MJD-MID", "MJDMID", "MJD-OBS", "MJD-BEG"):
            if k in h:
                try:
                    return Time(float(h[k]), format="mjd")
                except (TypeError, ValueError):
                    pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return Time(m.group(1))


def output_grid(ra0, dec0, size_deg):
    """
    North-up TAN grid centred on the target. The cutouts are squares in their own
    (rotated) orientation, so we show the largest square that fits inside any
    rotation: side = size / sqrt(2). That avoids empty corners.
    """
    side_arcsec = size_deg * 3600 / np.sqrt(2)
    n = int(side_arcsec / GRID_SCALE)
    n += (n % 2 == 0)  # odd, so the target sits on the central pixel
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [ra0, dec0]
    w.wcs.crpix = [(n + 1) / 2, (n + 1) / 2]
    w.wcs.cdelt = [-GRID_SCALE / 3600, GRID_SCALE / 3600]  # east to the left, as on the sky
    return w, (n, n)


def load_aligned(path, grid_wcs, shape):
    with fits.open(path) as h:
        img = h["IMAGE"].data.astype(float)
        if "ZODI" in h:
            img = img - h["ZODI"].data.astype(float)
        flags = (h["FLAGS"].data != 0).astype(float) if "FLAGS" in h else None
        wcs = WCS(h["IMAGE"].header).celestial
        t = obs_time(path, [h["IMAGE"].header, h[0].header])
    aligned, footprint = reproject_interp((img, wcs), grid_wcs, shape_out=shape)
    aligned[footprint == 0] = np.nan
    fl = None
    if flags is not None:
        fl, _ = reproject_interp((flags, wcs), grid_wcs, shape_out=shape, order="nearest-neighbor")
    missing = np.isnan(img).mean()
    return {"img": aligned, "flags": fl, "time": t, "footprint": footprint, "missing": float(missing)}


def group_epochs(years, gap_days=60):
    order = np.argsort(years)
    labels = np.zeros(len(years), dtype=int)
    cur = 0
    for a, b in zip(order[:-1], order[1:]):
        if (years[b] - years[a]) * 365.25 > gap_days:
            cur += 1
        labels[b] = cur
    return labels


# ---------------------------------------------------------------------------
# Pictures
# ---------------------------------------------------------------------------
def to_gray(arr, lo, hi, nan_value=255):
    """asinh stretch to 0-255; missing pixels (saturated star cores) shown white."""
    a = 10.0
    x = np.clip((arr - lo) / (hi - lo), 0, 1)
    g = (np.arcsinh(a * x) / np.arcsinh(a) * 255)
    g = np.where(np.isfinite(arr), g, nan_value).astype(np.uint8)
    return np.flipud(g)  # FITS row 0 is at the bottom; PNG row 0 is at the top


def save_png(gray_or_rgb, path):
    im = Image.fromarray(gray_or_rgb)
    im = im.resize((im.width * UPSCALE, im.height * UPSCALE), Image.NEAREST)
    im.save(path)
    return im


def diff_rgb(diff, lim):
    cmap = matplotlib.colormaps["RdBu_r"]
    x = np.clip((diff / lim + 1) / 2, 0, 1)
    rgb = (cmap(np.nan_to_num(x, nan=0.5))[..., :3] * 255).astype(np.uint8)
    rgb[~np.isfinite(diff)] = (128, 128, 128)
    return np.flipud(rgb)


# ---------------------------------------------------------------------------
# Stacking one detector band into epochs
# ---------------------------------------------------------------------------
RGB_CHANNELS = {"R": "D6", "G": "D3", "B": "D1"}  # long wavelength = red, short = blue


def _band_stacks(target, ra, dec, band, size, data_dir, max_per_band, fetch):
    """Load, align and stack one detector band. Returns a dict with per-epoch stacks."""
    key, label, ra0, dec0 = resolve(target, ra, dec)
    latest = None
    if fetch:
        folder, latest = ensure_data(target, ra, dec, bands=(band,), size=size,
                                     max_per_band=max_per_band, data_dir=data_dir)
    else:
        folder = data_folder(label, size, data_dir)
    files = sorted(folder.glob(f"*_{band}_*.fits"))
    if not files:
        raise FileNotFoundError(f"No SPHEREx {band} images found for {label}.")

    grid, shape = output_grid(ra0, dec0, size)
    loaded = []
    for f in files:
        try:
            loaded.append(load_aligned(f, grid, shape))
        except Exception as exc:
            print(f"[pipeline] skipped {f.name}: {type(exc).__name__}: {str(exc)[:80]}")
    if not loaded:
        raise RuntimeError(f"Could not load any {band} images.")

    years = np.array([d["time"].decimalyear for d in loaded])
    labels = group_epochs(years)
    epochs = []
    for e in np.unique(labels):
        members = [d for d, l in zip(loaded, labels) if l == e]
        cube = np.array([d["img"] for d in members])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pixels are expected
            stack = np.nanmedian(cube, axis=0)
            fl = [d["flags"] for d in members if d["flags"] is not None]
            flag_frac = float(np.nanmean(np.array(fl))) if fl else None
        year = float(np.mean([d["time"].decimalyear for d in members]))
        epochs.append({"stack": stack, "n": len(members), "year": year,
                       "date": Time(year, format="decimalyear").iso[:10],
                       "flag_fraction": flag_frac,
                       "missing_fraction": float(np.mean([d["missing"] for d in members]))})
    return {"key": key, "label": label, "ra0": ra0, "dec0": dec0, "shape": shape,
            "n_files": len(files), "latest": latest, "epochs": epochs}


def _stretch_limits(stacks):
    finite = np.concatenate([s[np.isfinite(s)].ravel() for s in stacks])
    lo, hi = np.percentile(finite, [5, 99.7])
    return float(lo), float(hi if hi > lo else lo + 1)


def _manifest(key, label, ra0, dec0, shape, band, size, url, frames, n_files, latest, extra=None):
    m = {
        "target": {"key": key, "label": label, "ra": ra0, "dec": dec0},
        "band": band,
        "size_deg": size,
        "grid": {"nx": shape[1], "ny": shape[0], "scale_arcsec": GRID_SCALE, "upscale": UPSCALE,
                 "ra0": ra0, "dec0": dec0,
                 "note": "PNG pixel (px,py), top-left origin: "
                         "dx=(px/upscale - (nx-1)/2), dy=((ny-1)/2 - py/upscale); "
                         "ra = ra0 - dx*scale/3600/cos(dec0); dec = dec0 + dy*scale/3600"},
        "data_folder": data_folder(label, size).name,
        "frames": frames,
        "deep_url": f"{url}/deep.png",
        "diff_url": f"{url}/diff.png",
        "blink_url": f"{url}/blink.gif",
        "latest_image_date": latest or max(f["date"] for f in frames),
        "n_images_total": n_files,
        "n_epochs": len(frames),
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    if extra:
        m.update(extra)
    return m


def _blink(first, last, path):
    h, w = first.shape[:2]
    a = Image.fromarray(first).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)
    b = Image.fromarray(last).resize((w * UPSCALE, h * UPSCALE), Image.NEAREST)
    if a.mode == "RGB":  # GIFs need a palette
        a, b = a.quantize(256), b.quantize(256)
    a.save(path, save_all=True, append_images=[b], duration=600, loop=0)


# ---------------------------------------------------------------------------
# Main entry point (used by main.py and the MCP server)
# ---------------------------------------------------------------------------
def build_frames(target=None, ra=None, dec=None, band="D1", size=DEFAULT_SIZE,
                 data_dir="data", max_per_band=40, fetch=True, force=False):
    band = band.upper()
    key, label, ra0, dec0 = resolve(target, ra, dec)
    out = FRAMES_DIR / f"{label}_{band}_{size:g}"
    manifest_path = out / "manifest.json"
    if (manifest_path.exists() and not force and
            time.time() - manifest_path.stat().st_mtime < CACHE_HOURS * 3600):
        return json.loads(manifest_path.read_text())  # cached frames: instant, no IRSA needed
    if band == "RGB":
        return build_rgb(target, ra, dec, size, data_dir, max_per_band, fetch)

    s = _band_stacks(target, ra, dec, band, size, data_dir, max_per_band, fetch)
    epochs, shape = s["epochs"], s["shape"]
    out.mkdir(parents=True, exist_ok=True)

    # One brightness scale for every frame, so changes are real changes
    lo, hi = _stretch_limits([e["stack"] for e in epochs])
    url = f"/frames/{out.name}"
    frames, grays = [], []
    for i, e in enumerate(epochs, 1):
        g = to_gray(e["stack"], lo, hi)
        grays.append(g)
        save_png(g, out / f"e{i}.png")
        frames.append({"epoch": i, "date": e["date"], "band": band, "n_images": e["n"],
                       "url": f"{url}/e{i}.png",
                       "flag_fraction": None if e["flag_fraction"] is None else round(e["flag_fraction"], 3),
                       "missing_fraction": round(e["missing_fraction"], 4)})

    # Deep image: all epochs together (reveals faint objects)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        deep = np.nanmedian(np.array([e["stack"] for e in epochs]), axis=0)
    save_png(to_gray(deep, lo, hi), out / "deep.png")

    # Difference: last minus first (movers show as a red/blue pair)
    diff = epochs[-1]["stack"] - epochs[0]["stack"]
    lim = np.nanpercentile(np.abs(diff), 99.5) if np.isfinite(diff).any() else 1.0
    save_png(diff_rgb(diff, lim or 1.0), out / "diff.png")

    _blink(grays[0], grays[-1], out / "blink.gif")

    m = _manifest(s["key"], s["label"], s["ra0"], s["dec0"], shape, band, size, url, frames,
                  s["n_files"], s["latest"])
    manifest_path.write_text(json.dumps(m, indent=2))
    return m


def build_rgb(target=None, ra=None, dec=None, size=DEFAULT_SIZE, data_dir="data",
              max_per_band=40, fetch=True):
    """
    False-colour view: D6 (4.4-5.0 um) = red, D3 (1.6-2.4 um) = green, D1 (0.75-1.1 um) = blue.
    Each colour is scaled separately, so an object's colour shows where in the infrared it is
    brightest: cold brown dwarfs (bright at long wavelengths) look red, normal stars white/blue.
    """
    stacks, missing = {}, []
    for ch, band in RGB_CHANNELS.items():
        try:
            stacks[ch] = _band_stacks(target, ra, dec, band, size, data_dir, max_per_band, fetch)
        except (FileNotFoundError, RuntimeError) as exc:
            missing.append(f"{band} ({exc})")
    if missing:
        raise FileNotFoundError("RGB needs detectors D1, D3 and D6. Missing: " + "; ".join(missing))

    ref = stacks["G"]
    shape = ref["shape"]
    # Match epochs across bands: for each D3 epoch, the nearest D1 and D6 epoch within 90 days
    matched = []
    for e in ref["epochs"]:
        trio = {"G": e}
        for ch in ("R", "B"):
            best = min(stacks[ch]["epochs"], key=lambda o: abs(o["year"] - e["year"]))
            if abs(best["year"] - e["year"]) * 365.25 <= 90:
                trio[ch] = best
        if len(trio) == 3:
            matched.append(trio)
    if not matched:
        raise FileNotFoundError("The D1, D3 and D6 images were taken in different sky passes; no common epoch.")

    limits = {ch: _stretch_limits([e["stack"] for e in stacks[ch]["epochs"]]) for ch in RGB_CHANNELS}
    out = FRAMES_DIR / f"{ref['label']}_RGB_{size:g}"
    out.mkdir(parents=True, exist_ok=True)
    url = f"/frames/{out.name}"

    def rgb_image(planes):
        chans = [to_gray(planes[ch], *limits[ch]) for ch in ("R", "G", "B")]
        return np.dstack(chans)

    frames, images = [], []
    for i, trio in enumerate(matched, 1):
        img = rgb_image({ch: trio[ch]["stack"] for ch in RGB_CHANNELS})
        images.append(img)
        save_png(img, out / f"e{i}.png")
        frames.append({"epoch": i, "date": trio["G"]["date"], "band": "RGB",
                       "n_images": int(sum(trio[ch]["n"] for ch in RGB_CHANNELS)),
                       "url": f"{url}/e{i}.png", "flag_fraction": None,
                       "missing_fraction": round(float(np.mean([trio[ch]["missing_fraction"]
                                                                for ch in RGB_CHANNELS])), 4)})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        deep = {ch: np.nanmedian(np.array([e["stack"] for e in stacks[ch]["epochs"]]), axis=0)
                for ch in RGB_CHANNELS}
    save_png(rgb_image(deep), out / "deep.png")

    # Difference on the combined (normalised) brightness of the three bands
    def norm(a, ch):
        lo, hi = limits[ch]
        return (a - lo) / (hi - lo)
    first = sum(norm(matched[0][ch]["stack"], ch) for ch in RGB_CHANNELS)
    last = sum(norm(matched[-1][ch]["stack"], ch) for ch in RGB_CHANNELS)
    diff = last - first
    lim = np.nanpercentile(np.abs(diff), 99.5) if np.isfinite(diff).any() else 1.0
    save_png(diff_rgb(diff, lim or 1.0), out / "diff.png")

    _blink(images[0], images[-1], out / "blink.gif")

    n_files = sum(stacks[ch]["n_files"] for ch in RGB_CHANNELS)
    latest = max((stacks[ch]["latest"] for ch in RGB_CHANNELS if stacks[ch]["latest"]), default=None)
    m = _manifest(ref["key"], ref["label"], ref["ra0"], ref["dec0"], shape, "RGB", size, url, frames,
                  n_files, latest, extra={"channels": {"red": "D6 (4.4-5.0 um)", "green": "D3 (1.6-2.4 um)",
                                                        "blue": "D1 (0.75-1.1 um)"}})
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def main():
    ap = argparse.ArgumentParser(description="Build aligned SPHEREx frames for the website")
    ap.add_argument("--target")
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--band", default="D1", help="D1..D6, or RGB (D6/D3/D1 as red/green/blue)")
    ap.add_argument("--size", type=float, default=DEFAULT_SIZE, help="cutout size in degrees")
    ap.add_argument("--max", type=int, default=40, help="max images per band to download")
    ap.add_argument("--no-fetch", action="store_true", help="use only files already downloaded")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    m = build_frames(args.target, args.ra, args.dec, args.band, args.size,
                     max_per_band=args.max, fetch=not args.no_fetch, force=args.force)
    print(f"\n{m['target']['label']} {m['band']}: {m['n_images_total']} images -> {m['n_epochs']} epochs")
    for f in m["frames"]:
        print(f"  epoch {f['epoch']}: {f['date']}  n={f['n_images']:2d}  flagged={f['flag_fraction']}")
    out = FRAMES_DIR / f"{m['target']['label']}_{m['band']}_{m['size_deg']:g}"
    print(f"\nSaved frames, deep.png, diff.png, blink.gif and manifest.json in {out}")


if __name__ == "__main__":
    main()
