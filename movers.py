"""
movers.py - scan a field for moving objects and check each one against Gaia.

How it works:
  1. Use the aligned epoch stacks from pipeline.py (one clean image per six-month pass).
  2. Find every source in the deep image (all epochs together).
  3. Fit each source's position in every epoch stack -> motion in arcsec/yr.
  4. Sources that clearly move are looked up live in Gaia DR3:
       known mover (Gaia agrees)  /  DISCOVERY CANDIDATE (not in Gaia, or no Gaia motion).

Usage:
    python movers.py --target barnard --band D3
    python movers.py --ra 269.4463 --dec 4.7677 --band D2
API: /api/movers?target=barnard&band=D3
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.ndimage import maximum_filter

import pipeline
from analysis import _gaia_cone, gaia_lookup, psf_fit
from fetch_spherex import PRESETS

MIN_SNR = 6.0          # detection threshold in the deep image
MIN_MOTION = 1.5       # arcsec/yr: slower than this is not reported as a mover
SIGMA_CUT = 3.0        # motion must be this many sigma from zero
MAX_SOURCES = 60       # brightest sources examined
MAX_GAIA = 12          # Gaia lookups per scan (each takes a few seconds)


def _noise(img):
    v = img[np.isfinite(img)]
    med = np.median(v)
    return med, 1.4826 * np.median(np.abs(v - med))


def _detect(deep, border=4):
    """Local maxima above MIN_SNR. Saturated cores (NaN) count as very bright."""
    bg, sig = _noise(deep)
    work = deep - bg
    nan = ~np.isfinite(work)
    work = np.where(nan, np.nanmax(work) if np.isfinite(work).any() else 0, work)
    peaks = (work == maximum_filter(work, size=5)) & (work > MIN_SNR * sig)
    peaks[:border, :] = peaks[-border:, :] = False
    peaks[:, :border] = peaks[:, -border:] = False
    ys, xs = np.nonzero(peaks)
    order = np.argsort(-work[ys, xs])[:MAX_SOURCES]
    return [(float(xs[i]), float(ys[i]), float(work[ys[i], xs[i]] / sig)) for i in order], sig


def _position_in(stack, x0, y0, search=3):
    """Brightest pixel near (x0, y0), refined by a Gaussian fit that ignores NaN pixels."""
    ny, nx = stack.shape
    xi, yi = int(round(x0)), int(round(y0))
    y1, y2 = max(yi - search, 0), min(yi + search + 1, ny)
    x1, x2 = max(xi - search, 0), min(xi + search + 1, nx)
    win = stack[y1:y2, x1:x2]
    if not np.isfinite(win).any():
        return None
    filled = np.where(np.isfinite(win), win, np.nanmax(win))  # saturated hole = brightest
    dy, dx = np.unravel_index(np.argmax(filled), win.shape)
    xb, yb = x1 + dx, y1 + dy
    fit = psf_fit(stack, float(xb), float(yb), half=4)
    if fit is None:
        return None
    bg, sig = _noise(stack)
    peak = np.nanmax(filled)
    snr = (peak - bg) / sig if sig > 0 else 0
    return fit[0], fit[1], float(snr)


def _wfit(t, v, err):
    w = 1.0 / np.asarray(err) ** 2
    A = np.vstack([t, np.ones_like(t)]).T
    cov = np.linalg.inv(A.T @ (A * w[:, None]))
    p = cov @ (A.T @ (w * v))
    return float(p[0]), float(np.sqrt(cov[0, 0]))


def scan(target=None, ra=None, dec=None, band="D1", size=pipeline.DEFAULT_SIZE, fetch=True,
         check_gaia=True, force=False):
    band = band.upper()
    key, label, ra0, dec0 = pipeline.resolve(target, ra, dec)
    out_dir = pipeline.FRAMES_DIR / f"{label}_{band}_{size:g}"
    cache = out_dir / "movers.json"
    if cache.exists() and not force and time.time() - cache.stat().st_mtime < pipeline.CACHE_HOURS * 3600:
        return json.loads(cache.read_text())

    s = pipeline._band_stacks(target, ra, dec, band, size, "data", 40, fetch)
    epochs = s["epochs"]
    if len(epochs) < 2:
        raise ValueError("Need at least 2 epochs (sky passes) to look for moving objects.")
    grid, shape = pipeline.output_grid(s["ra0"], s["dec0"], size)
    ny, nx = shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        deep = np.nanmedian(np.array([e["stack"] for e in epochs]), axis=0)
    sources, _ = _detect(deep)

    scale = pipeline.GRID_SCALE
    years = np.array([e["year"] for e in epochs])
    tmid = float(years.mean())
    results = []
    for x0, y0, snr_deep in sources:
        pts = []
        for e, yr in zip(epochs, years):
            p = _position_in(e["stack"], x0, y0)
            if p is not None:
                pts.append((yr - tmid, p[0], p[1], p[2]))
        if len(pts) < 2:
            continue
        t = np.array([p[0] for p in pts])
        # east is to the LEFT on the grid (x decreases), north is up (y increases)
        east = -np.array([p[1] for p in pts]) * scale
        north = np.array([p[2] for p in pts]) * scale
        err = np.array([max(0.3, 6.0 / max(p[3], 1e-3)) for p in pts])  # arcsec
        pmra, pmra_err = _wfit(t, east, err)
        pmdec, pmdec_err = _wfit(t, north, err)
        total = float(np.hypot(pmra, pmdec))
        total_err = float(np.hypot(pmra * pmra_err, pmdec * pmdec_err) / total) if total > 0 else 1.0
        xm, ym = float(np.mean([p[1] for p in pts])), float(np.mean([p[2] for p in pts]))
        sra, sdec = grid.pixel_to_world_values(xm, ym)
        moving = total > MIN_MOTION and total > SIGMA_CUT * total_err
        results.append({
            "ra": round(float(sra), 6), "dec": round(float(sdec), 6),
            "x_png": round((xm + 0.5) * pipeline.UPSCALE, 1),
            "y_png": round((ny - 1 - ym + 0.5) * pipeline.UPSCALE, 1),
            "snr": round(snr_deep, 1), "n_epochs": len(pts),
            "pmra": round(pmra, 2), "pmdec": round(pmdec, 2),
            "total": round(total, 2), "total_err": round(total_err, 2),
            "moving": bool(moving), "status": "static",
        })

    movers = sorted([r for r in results if r["moving"]], key=lambda r: -r["snr"])

    # Is a mover the (already known) target itself? Then say so instead of "discovery".
    target_pos = None
    if key:
        p = PRESETS[key]
        dt = tmid - p["ref"]
        tdec = p["dec"] + p["pmdec"] * dt / 3.6e6
        tra = p["ra"] + p["pmra"] * dt / 3.6e6 / np.cos(np.radians(tdec))
        target_pos = (tra, tdec, p["label"].replace("_", " "))

    def is_target(r):
        if target_pos is None:
            return False
        sep = np.hypot((r["ra"] - target_pos[0]) * np.cos(np.radians(r["dec"])), r["dec"] - target_pos[1]) * 3600
        return sep < 12

    gaia_ok = True
    for r in movers[:MAX_GAIA] if check_gaia else []:
        try:
            g = gaia_lookup(r["ra"], r["dec"], tmid)
        except Exception as exc:
            gaia_ok = False
            r["status"] = "unverified (Gaia offline)"
            r["gaia"] = f"{type(exc).__name__}"
            continue
        if g is None or not g["has_pm"]:
            why = "no Gaia DR3 star here" if g is None else f"Gaia DR3 {g['source_id']} has no measured motion"
            if is_target(r):
                r["status"] = f"known target: {target_pos[2]}"
                r["gaia"] = why + " (too faint in visible light for Gaia; SPHEREx sees it in the infrared)"
            else:
                r["status"] = "DISCOVERY CANDIDATE"
                r["gaia"] = why
        else:
            gt = float(np.hypot(g["pmra"], g["pmdec"]))
            chi = np.hypot((r["pmra"] - g["pmra"]) / max(r["total_err"], 0.3),
                           (r["pmdec"] - g["pmdec"]) / max(r["total_err"], 0.3))
            r["status"] = "known mover" if chi < 3 else "known star, motion differs"
            r["gaia"] = f"Gaia DR3 {g['source_id']} (G={g['gmag']}): {gt:.2f}\"/yr"
    for r in movers[MAX_GAIA:]:
        r["status"] = "moving (not checked)"
    for r in movers:  # label the target even when Gaia could not be checked
        if is_target(r) and not r["status"].startswith("known"):
            r["status"] = f"known target: {target_pos[2]}" + ("" if r["status"] == "static" else "")

    result = {
        "target": {"key": key, "label": label, "ra": s["ra0"], "dec": s["dec0"]},
        "band": band, "size_deg": size,
        "png_size": [nx * pipeline.UPSCALE, ny * pipeline.UPSCALE],
        "n_sources": len(results), "n_movers": len(movers),
        "n_candidates": sum(1 for r in movers if r["status"] == "DISCOVERY CANDIDATE"),
        "gaia_checked": bool(check_gaia and gaia_ok),
        "sources": sorted(results, key=lambda r: (not r["moving"], -r["snr"])),
        "settings": {"min_snr": MIN_SNR, "min_motion_arcsec_yr": MIN_MOTION, "sigma_cut": SIGMA_CUT},
        "note": "Candidates need checking by eye (Blink/Difference/WiseView): blends, artefacts and "
                "saturated neighbours can mimic motion.",
        "generated": time.strftime("%Y-%m-%d %H:%M"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if gaia_ok:  # don't cache a result made while Gaia was offline
        cache.write_text(json.dumps(result, indent=2))
    return result


def gaia_overlay(target=None, ra=None, dec=None, band="D1", size=pipeline.DEFAULT_SIZE, fetch=True,
                 max_stars=300):
    """
    Gaia DR3 stars in the field of a SPHEREx view, moved along their proper motions to the
    middle of the SPHEREx observations, with their pixel positions on the website's frames.
    Uses the Gaia mirrors in analysis.py (ESA, ARI Heidelberg, GAVO, NOIRLab), not CDS.
    """
    m = pipeline.build_frames(target, ra, dec, band, size, fetch=fetch)
    g = m["grid"]
    ra0, dec0, scale, up = g["ra0"], g["dec0"], g["scale_arcsec"], g["upscale"]
    nx, ny = g["nx"], g["ny"]
    years = []
    for f in m["frames"]:
        y, mo, d = (int(v) for v in f["date"].split("-"))
        years.append(y + (mo - 1) / 12 + (d - 1) / 365.25)
    tmid = float(np.mean(years))
    half_diag = np.hypot(nx, ny) / 2 * scale            # arcsec, covers the whole frame
    rows, service = _gaia_cone(ra0, dec0, half_diag + 30)  # margin: stars move into the field
    cosd = np.cos(np.radians(dec0))
    stars = []
    for r in rows:
        if r["ra"] is None or r["dec"] is None:
            continue
        has_pm = r["pmra"] is not None and r["pmdec"] is not None
        pmra = r["pmra"] if has_pm else 0.0
        pmdec = r["pmdec"] if has_pm else 0.0
        dt = tmid - 2016.0
        sdec = r["dec"] + pmdec * dt / 3.6e6
        sra = r["ra"] + pmra * dt / 3.6e6 / np.cos(np.radians(sdec))
        dx = -(sra - ra0) * cosd * 3600 / scale      # east is to the left
        dy = (sdec - dec0) * 3600 / scale
        x = (nx - 1) / 2 + dx
        y = (ny - 1) / 2 + dy
        if not (-0.5 <= x <= nx - 0.5 and -0.5 <= y <= ny - 0.5):
            continue
        stars.append({
            "source_id": r["source_id"], "ra": round(sra, 6), "dec": round(sdec, 6),
            "gmag": None if r["gmag"] is None else round(r["gmag"], 2),
            "pm_total": round(float(np.hypot(pmra, pmdec)) / 1000, 3) if has_pm else None,
            "x_png": round((x + 0.5) * up, 1), "y_png": round((ny - 1 - y + 0.5) * up, 1),
        })
    stars.sort(key=lambda s: 99 if s["gmag"] is None else s["gmag"])
    return {"service": service, "epoch": round(tmid, 3), "n_stars": len(stars),
            "png_size": [nx * up, ny * up], "stars": stars[:max_stars],
            "note": "Gaia DR3 positions moved to the mean SPHEREx epoch using Gaia proper motions."}


def main():
    ap = argparse.ArgumentParser(description="Scan a SPHEREx field for moving objects")
    ap.add_argument("--target")
    ap.add_argument("--ra", type=float)
    ap.add_argument("--dec", type=float)
    ap.add_argument("--band", default="D1")
    ap.add_argument("--size", type=float, default=pipeline.DEFAULT_SIZE)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--no-gaia", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    r = scan(a.target, a.ra, a.dec, a.band, a.size, fetch=not a.no_fetch,
             check_gaia=not a.no_gaia, force=a.force)
    print(f"\n{r['target']['label']} {r['band']}: {r['n_sources']} sources, {r['n_movers']} moving, "
          f"{r['n_candidates']} discovery candidates")
    for s in r["sources"]:
        if s["moving"]:
            print(f"  RA {s['ra']:.5f} Dec {s['dec']:+.5f}  {s['total']:.2f}±{s['total_err']:.2f}\"/yr  "
                  f"S/N {s['snr']:.0f}  -> {s['status']}  {s.get('gaia', '')}")


if __name__ == "__main__":
    main()
