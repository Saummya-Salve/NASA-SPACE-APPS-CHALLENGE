"""
analysis.py  (Person 2: Science)

Measures how fast a star moves across the sky using SPHEREx cutouts, and
compares the result with the known (reference) proper motion.

It works directly on the raw cutouts that fetch_spherex.py downloads, so you
do NOT need Person 1's pipeline.py: every cutout carries its own WCS (the map
from pixel to sky position), so we can measure the star's sky position in each
image separately, even though the images are rotated differently.

Usage (from the project folder, .venv active):

    python analysis.py motion --target barnard --band D1
    python analysis.py lightcurve --target rdra --band D1

Outputs:
    - a summary printed in the terminal
    - data/<target>/motion_<band>.json   (same shape as the /api/motion contract)
    - data/<target>/motion_<band>.png    (diagnostic plot for the pitch)
    - data/<target>/lightcurve_<band>.json / .png  (for variable stars)

First download data with, e.g.:
    python fetch_spherex.py --name "Barnard's Star" --band D1
"""

import argparse
import json
import re
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import log as astropy_log
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
from astropy.wcs import WCS, FITSFixedWarning
from astropy.wcs.utils import proj_plane_pixel_area

from fetch_spherex import ALIASES, PRESETS

warnings.simplefilter("ignore", FITSFixedWarning)
astropy_log.setLevel("WARNING")  # hide the harmless SIP INFO messages from the wavelength WCS

# ---------------------------------------------------------------------------
# Reference values (TASK 1: confirm these against the Gaia archive!)
# pm in mas/yr. These mirror the positions built into fetch_spherex.py.
# ---------------------------------------------------------------------------
REFERENCE_SOURCE = {
    "barnard": "Gaia DR3",
    "epsindib": "published value for the Epsilon Indi system; Epsilon Indi B itself is too faint in "
                "visible light to be in Gaia",
    "wise0855": "published value (Spitzer); WISE 0855 is far too faint in visible light for Gaia",
    "rwaur": "variable star; motion not tested",
    "rdra": "variable star; motion not tested",
}


def resolve_target(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key not in ALIASES:
        raise SystemExit(f"Unknown target '{name}'. Use one of: {', '.join(PRESETS)}")
    return ALIASES[key]


# ---------------------------------------------------------------------------
# Loading cutouts
# ---------------------------------------------------------------------------
def obs_time(headers, path: Path) -> Time:
    """Best available observation time from the FITS headers, else the filename date."""
    for h in headers:
        for k in ("MJD-AVG", "MJD-MID", "MJDMID", "MJD-OBS", "MJD-BEG"):
            if k in h:
                try:
                    return Time(float(h[k]), format="mjd")
                except (TypeError, ValueError):
                    pass
        for k in ("DATE-AVG", "DATE-OBS", "DATE-BEG"):
            if k in h:
                try:
                    return Time(h[k])
                except (TypeError, ValueError):
                    pass
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    if not m:
        raise ValueError(f"No observation time found for {path.name}")
    return Time(m.group(1))


def load_cutout(path: Path):
    with fits.open(path) as hdul:
        hdr = hdul["IMAGE"].header
        img = hdul["IMAGE"].data.astype(float)
        if "ZODI" in hdul:
            img = img - hdul["ZODI"].data.astype(float)
        flags = hdul["FLAGS"].data.copy() if "FLAGS" in hdul else None
        wcs = WCS(hdr).celestial
        t = obs_time([hdr, hdul[0].header], path)
    return {"path": path, "img": img, "flags": flags, "wcs": wcs, "time": t}


def find_files(key: str, band: str, data_dir: Path):
    folder = data_dir / PRESETS[key]["label"]
    files = sorted(folder.glob(f"*_{band.upper()}_*.fits"))
    if not files:
        raise SystemExit(
            f"No {band} cutouts in {folder}.\n"
            f"Download them first:  python fetch_spherex.py --name \"{key}\" --band {band}"
        )
    return folder, files


# ---------------------------------------------------------------------------
# Measuring a star's position in one image
# ---------------------------------------------------------------------------
def predicted_radec(key: str, t: Time):
    """Where the reference proper motion says the star should be at time t."""
    p = PRESETS[key]
    dt = t.decimalyear - p["ref"]
    dec = p["dec"] + p["pmdec"] * dt / 3.6e6
    ra = p["ra"] + p["pmra"] * dt / 3.6e6 / np.cos(np.radians(dec))
    return ra, dec


def centroid(img, x0, y0, search=3, r=2, bg_r=7):
    """
    Lock on to the brightest pixel within `search` px of (x0, y0), then compute a
    background-subtracted, brightness-weighted centroid in a (2r+1)^2 box.
    Returns (x, y, snr) or None.
    """
    ny, nx = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    if not (bg_r <= xi < nx - bg_r and bg_r <= yi < ny - bg_r):
        return None  # too close to the edge of the cutout

    # 1. brightest pixel near the prediction
    win = img[yi - search:yi + search + 1, xi - search:xi + search + 1]
    if not np.isfinite(win).any():
        return None
    dy, dx = np.unravel_index(np.nanargmax(win), win.shape)
    xi, yi = xi - search + dx, yi - search + dy
    if not (bg_r <= xi < nx - bg_r and bg_r <= yi < ny - bg_r):
        return None

    # 2. background from a ring around the star
    big = img[yi - bg_r:yi + bg_r + 1, xi - bg_r:xi + bg_r + 1].copy()
    big[bg_r - r - 1:bg_r + r + 2, bg_r - r - 1:bg_r + r + 2] = np.nan
    bg = np.nanmedian(big)
    noise = 1.4826 * np.nanmedian(np.abs(big - bg))  # robust standard deviation

    # 3. weighted centroid
    box = img[yi - r:yi + r + 1, xi - r:xi + r + 1] - bg
    box = np.where(np.isfinite(box), np.clip(box, 0, None), 0.0)
    total = box.sum()
    if total <= 0:
        return None
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    x = xi + (box * xx).sum() / total
    y = yi + (box * yy).sum() / total
    snr = (img[yi, xi] - bg) / noise if noise > 0 else np.inf
    return x, y, float(snr)


def psf_fit(img, x0, y0, half=4, bad=None):
    """
    Fit a 2-D Gaussian + flat background to the star, IGNORING missing pixels and
    any pixels marked in `bad` (e.g. saturated core). For saturated stars the
    unsaturated wings still pin down the centre, whereas a simple centroid gets
    pulled sideways by the hole in the middle.
    Returns (x, y) or None.
    """
    from scipy.optimize import least_squares

    ny, nx = img.shape
    xi, yi = int(round(x0)), int(round(y0))
    if not (half <= xi < nx - half and half <= yi < ny - half):
        return None
    sub = img[yi - half:yi + half + 1, xi - half:xi + half + 1]
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    xx = xx + xi
    yy = yy + yi
    ok = np.isfinite(sub)
    if bad is not None:
        ok &= ~bad[yi - half:yi + half + 1, xi - half:xi + half + 1]
    if ok.sum() < 12:
        return None
    bg0 = np.nanmedian(sub[~np.isfinite(sub) | (sub < np.nanpercentile(sub, 50))]) if ok.any() else 0.0
    amp0 = max(np.nanmax(np.where(ok, sub, np.nan)) - bg0, 1e-6) * 2.0

    def resid(p):
        a, x, y, s, b = p
        model = b + a * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * s ** 2))
        return (model - sub)[ok]

    p0 = [amp0, x0, y0, 1.0, bg0]
    lo = [0, x0 - 2, y0 - 2, 0.4, -np.inf]
    hi = [np.inf, x0 + 2, y0 + 2, 4.0, np.inf]
    try:
        r = least_squares(resid, p0, bounds=(lo, hi))
    except Exception:
        return None
    if not r.success:
        return None
    return float(r.x[1]), float(r.x[2])


# ---------------------------------------------------------------------------
# Proper motion
# ---------------------------------------------------------------------------
def group_epochs(years, gap_days=60):
    """Split times into epochs (sky passes) wherever there's a gap > gap_days."""
    order = np.argsort(years)
    labels = np.zeros(len(years), dtype=int)
    current = 0
    for prev, cur in zip(order[:-1], order[1:]):
        if (years[cur] - years[prev]) * 365.25 > gap_days:
            current += 1
        labels[cur] = current
    return labels


def fit_line(t, v, clip=3.0, iters=3):
    """Straight-line fit with sigma clipping. Returns slope, slope_err, intercept, keep mask."""
    keep = np.ones(len(t), bool)
    for _ in range(iters):
        if keep.sum() < 4:
            break
        p = np.polyfit(t[keep], v[keep], 1)
        resid = v - np.polyval(p, t)
        s = 1.4826 * np.median(np.abs(resid[keep] - np.median(resid[keep])))
        if s == 0:
            break
        new = np.abs(resid) < clip * s
        if (new == keep).all():
            break
        keep = new
    if keep.sum() < 4:
        raise ValueError("Not enough good measurements to fit a motion (need at least 4).")
    p, cov = np.polyfit(t[keep], v[keep], 1, cov=True)
    return p[0], float(np.sqrt(cov[0, 0])), p[1], keep


# Gaia DR3 is mirrored at several data centres. Try them in turn (ESA was overloaded
# during the hackathon) and cache every answer on disk.
GAIA_SERVICES = [
    ("ESA Gaia archive", "https://gea.esac.esa.int/tap-server/tap", "gaiadr3.gaia_source"),
    ("ARI Heidelberg", "https://gaia.ari.uni-heidelberg.de/tap", "gaiadr3.gaia_source"),
    ("GAVO Data Center", "https://dc.g-vo.org/tap", "gaia.dr3lite"),
    ("NOIRLab Astro Data Lab", "https://datalab.noirlab.edu/tap", "gaia_dr3.gaia_source"),
]
GAIA_CACHE = Path("data") / "gaia_cache.json"


def _gaia_cone(ra, dec, radius_arcsec):
    """Return (rows, service_name). rows: list of dicts with source_id, ra, dec, pmra, pmdec, gmag."""
    import requests
    import pyvo

    key = f"{ra:.5f},{dec:.5f},{radius_arcsec:.0f}"
    cache = {}
    if GAIA_CACHE.exists():
        try:
            cache = json.loads(GAIA_CACHE.read_text())
        except ValueError:
            cache = {}
    if key in cache:
        return cache[key]["rows"], cache[key]["service"] + " (cached)"

    class _TimeoutSession(requests.Session):
        def request(self, *a, **k):
            k.setdefault("timeout", 25)
            return super().request(*a, **k)

    r_deg = radius_arcsec / 3600
    errors = []
    for name, url, table in GAIA_SERVICES:
        q = (f"SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag FROM {table} "
             f"WHERE 1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {ra}, {dec}, {r_deg}))")
        try:
            svc = pyvo.dal.TAPService(url, session=_TimeoutSession())
            t = svc.run_sync(q).to_table()
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}")
            continue
        col = {c.lower(): c for c in t.colnames}

        def val(row, c):
            v = row[col[c]] if c in col else None
            return None if v is None or np.ma.is_masked(v) else float(v)
        rows = [{"source_id": str(row[col["source_id"]]), "ra": val(row, "ra"), "dec": val(row, "dec"),
                 "pmra": val(row, "pmra"), "pmdec": val(row, "pmdec"), "gmag": val(row, "phot_g_mean_mag")}
                for row in t]
        cache[key] = {"rows": rows, "service": name}
        try:
            GAIA_CACHE.parent.mkdir(parents=True, exist_ok=True)
            GAIA_CACHE.write_text(json.dumps(cache))
        except OSError:
            pass
        return rows, name
    raise ConnectionError("No Gaia service reachable: " + "; ".join(errors))


def gaia_lookup(ra, dec, epoch_year, radius_arcsec=20.0, match_arcsec=6.0):
    """
    Find the Gaia DR3 star at (ra, dec) as seen at epoch_year.
    Gaia positions are for 2016.0, so each nearby star is moved along its own proper
    motion to epoch_year before matching (fast stars move tens of arcsec in 10 years).
    Returns a dict, or None if no Gaia star matches. Raises if no Gaia service answers.
    """
    rows, service = _gaia_cone(ra, dec, radius_arcsec)
    best, best_sep = None, None
    for r in rows:
        if r["ra"] is None or r["dec"] is None:
            continue
        has_pm = r["pmra"] is not None and r["pmdec"] is not None
        pmra = r["pmra"] if has_pm else 0.0
        pmdec = r["pmdec"] if has_pm else 0.0
        dt = epoch_year - 2016.0
        g_dec_t = r["dec"] + pmdec * dt / 3.6e6
        g_ra_t = r["ra"] + pmra * dt / 3.6e6 / np.cos(np.radians(g_dec_t))
        sep = np.hypot((g_ra_t - ra) * np.cos(np.radians(dec)), g_dec_t - dec) * 3600
        if best_sep is None or sep < best_sep:
            best_sep = sep
            best = {"source_id": r["source_id"], "pmra": pmra / 1000, "pmdec": pmdec / 1000,
                    "has_pm": bool(has_pm), "gmag": None if r["gmag"] is None else round(r["gmag"], 2),
                    "separation_arcsec": round(float(sep), 2), "service": service}
    if best is None or best_sep > match_arcsec:
        return None
    return best


def measure_motion(target: str = None, band: str = "D1", data_dir="data", min_snr=10.0, make_plot=True,
                   exclude_flagged=False, method="psf", ra=None, dec=None, folder=None):
    """
    Measure proper motion. Tries single images first (most precise); if the object is too
    faint for that, a built-in target is measured on the stacked epochs instead.
    """
    try:
        r = _measure_motion_single(target, band, data_dir, min_snr, make_plot,
                                   exclude_flagged, method, ra, dec, folder)
        # A weak single-image result (few epochs or huge error bars): see whether the
        # stacked epochs do better, and keep the more precise of the two.
        weak = r["n_epochs"] < 3 or r["measured"]["total_err"] > 2.0
        if weak and target and ra is None and dec is None:
            try:
                st = measure_motion_stacked(target, band, data_dir)
                if st["measured"]["total_err"] < r["measured"]["total_err"]:
                    st["note"] = (f"Single images gave a weak result ({r['measured']['total']:.1f} ± "
                                  f"{r['measured']['total_err']:.1f}\"/yr from {r['n_used']} images); the "
                                  f"stacked epochs are more precise. " + st["note"])
                    return st
            except SystemExit:
                pass
        return r
    except SystemExit as single_error:
        if target and ra is None and dec is None:
            try:
                r = measure_motion_stacked(target, band, data_dir)
                r["note"] = (f"Too faint for single images ({single_error}); measured on stacked "
                             f"six-month epochs instead. " + r["note"])
                return r
            except SystemExit as stacked_error:
                raise SystemExit(f"{single_error} Stacked epochs also failed: {stacked_error}")
        raise


def measure_motion_stacked(target, band="D6", data_dir="data", size=0.05, min_snr=4.0):
    """
    Motion of a faint built-in target from the stacked epochs (one deep image per sky pass).
    Each stack adds ~10-15 images, so objects invisible in one image become measurable.
    Fewer points than the single-image method, so the uncertainty is larger.
    """
    import pipeline
    from movers import _position_in

    key = resolve_target(target)
    s = pipeline._band_stacks(target, None, None, band.upper(), size, data_dir, 40, fetch=False)
    grid, _ = pipeline.output_grid(s["ra0"], s["dec0"], size)
    scale = pipeline.GRID_SCALE

    pts, skipped = [], []
    for e in s["epochs"]:
        ra_p, dec_p = predicted_radec(key, Time(e["year"], format="decimalyear"))
        x0, y0 = grid.world_to_pixel_values(ra_p, dec_p)
        p = _position_in(e["stack"], float(x0), float(y0), search=2)
        if p is None or p[2] < min_snr:
            skipped.append((e["date"], "not detected" if p is None else f"S/N {p[2]:.1f}"))
            continue
        sra, sdec = grid.pixel_to_world_values(p[0], p[1])
        pts.append({"date": e["date"], "year": e["year"], "ra": float(sra), "dec": float(sdec),
                    "snr": p[2], "n": e["n"]})
    if len(pts) < 2:
        raise SystemExit(f"Detected in only {len(pts)} stacked epoch(s) (need 2): {skipped}")

    years = np.array([p["year"] for p in pts])
    ra_a = np.array([p["ra"] for p in pts])
    dec_a = np.array([p["dec"] for p in pts])
    t0, ra0, dec0 = years.mean(), ra_a.mean(), dec_a.mean()
    x_as = (ra_a - ra0) * np.cos(np.radians(dec0)) * 3600
    y_as = (dec_a - dec0) * 3600
    err = np.array([max(0.5, 6.0 / max(p["snr"], 1e-3)) for p in pts])  # arcsec, from S/N

    def wfit(t, v, e):
        w = 1.0 / e ** 2
        A = np.vstack([t, np.ones_like(t)]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        p = cov @ (A.T @ (w * v))
        return float(p[0]), float(np.sqrt(cov[0, 0]))

    dt = years - t0
    pmra, pmra_err = wfit(dt, x_as, err)
    pmdec, pmdec_err = wfit(dt, y_as, err)
    total = float(np.hypot(pmra, pmdec))
    total_err = float(np.hypot(pmra * pmra_err, pmdec * pmdec_err) / total) if total > 0 else float("nan")

    p = PRESETS[key]
    ref_pmra, ref_pmdec = p["pmra"] / 1000, p["pmdec"] / 1000
    ref_total = float(np.hypot(ref_pmra, ref_pmdec))
    sigma = float(np.hypot((pmra - ref_pmra) / pmra_err, (pmdec - ref_pmdec) / pmdec_err))
    verdict = "consistent" if sigma < 3 else "inconsistent"
    if len(pts) == 2:
        verdict += " (only 2 epochs: weak test)"

    return {
        "target": key, "star": None, "band": band.upper(), "method": "stacked epochs",
        "measured_arcsec_yr": round(total, 3), "gaia_arcsec_yr": round(ref_total, 3),
        "positions": [{"date": q["date"], "ra": q["ra"], "dec": q["dec"]} for q in pts],
        "measured": {"pmra": round(pmra, 3), "pmra_err": round(pmra_err, 3), "pmdec": round(pmdec, 3),
                     "pmdec_err": round(pmdec_err, 3), "total": round(total, 3), "total_err": round(total_err, 3)},
        "reference": {"pmra": round(ref_pmra, 3), "pmdec": round(ref_pmdec, 3), "total": round(ref_total, 3),
                      "source": REFERENCE_SOURCE[key]},
        "reference_note": "", "moving": bool(total > 3 * total_err),
        "difference_arcsec_yr": round(total - ref_total, 3), "difference_sigma": round(sigma, 2),
        "verdict": verdict,
        "n_used": int(sum(q["n"] for q in pts)), "n_rejected_outliers": 0, "n_epochs": len(pts),
        "n_skipped": len(skipped), "skipped": [{"file": d, "reason": r} for d, r in skipped],
        "n_core_flagged": 0, "n_core_missing": 0,
        "epochs": [{"epoch": i + 1, "date": q["date"], "n_images": q["n"], "snr": round(q["snr"], 1),
                    "x_arcsec": round(float(x), 2), "y_arcsec": round(float(y), 2),
                    "x_err": round(float(e), 2), "y_err": round(float(e), 2)}
                   for i, (q, x, y, e) in enumerate(zip(pts, x_as, y_as, err))],
        "baseline_years": round(float(years.max() - years.min()), 3),
        "note": "Positions measured on stacked images, one per six-month sky pass.",
    }


def _measure_motion_single(target: str = None, band: str = "D1", data_dir="data", min_snr=10.0,
                           make_plot=True, exclude_flagged=False, method="psf", ra=None, dec=None,
                           folder=None):
    """
    Measure proper motion from SPHEREx cutouts.
      - target only: one of the built-in targets (reference value built in)
      - ra/dec (+ target or folder): ANY star in that field; the reference comes
        from a live Gaia DR3 lookup. No Gaia match + clear motion = discovery candidate.
    """
    any_star = ra is not None and dec is not None
    key = resolve_target(target) if target else None
    if folder is None:
        if key is None:
            raise SystemExit("Give a target, or a data folder together with ra/dec.")
        folder = Path(data_dir) / PRESETS[key]["label"]
    folder = Path(folder)
    files = sorted(folder.glob(f"*_{band.upper()}_*.fits"))
    if not files:
        raise SystemExit(f"No {band} cutouts in {folder}.")

    def predict(t):
        return (ra, dec) if any_star else predicted_radec(key, t)

    rows, skipped = [], []
    for f in files:
        c = load_cutout(f)
        ra_p, dec_p = predict(c["time"])
        x0, y0 = c["wcs"].world_to_pixel_values(ra_p, dec_p)
        res = centroid(c["img"], float(x0), float(y0))
        if res is None:
            skipped.append((f.name, "no star / edge"))
            continue
        x, y, snr = res
        if snr < min_snr:
            skipped.append((f.name, f"low S/N {snr:.1f}"))
            continue
        # Is the star's core damaged (saturated / flagged / missing pixels)?
        xi, yi = int(round(x)), int(round(y))
        core = c["img"][yi - 1:yi + 2, xi - 1:xi + 2]
        core_nan = int((~np.isfinite(core)).sum())
        core_flag = int((c["flags"][yi - 1:yi + 2, xi - 1:xi + 2] != 0).sum()) if c["flags"] is not None else 0
        if exclude_flagged and (core_nan > 0 or core_flag > 0):
            skipped.append((f.name, f"core flagged ({core_flag} flagged, {core_nan} missing pixels)"))
            continue
        # Refine the position with a PSF fit that ignores missing and flagged core pixels
        if method == "psf":
            bad = None
            if c["flags"] is not None and (core_nan > 0):
                # star is saturated: ignore flagged pixels within 1.5 px of the centre
                yy_, xx_ = np.indices(c["img"].shape)
                bad = (c["flags"] != 0) & (np.hypot(xx_ - x, yy_ - y) <= 1.5)
            fit = psf_fit(c["img"], x, y, bad=bad)
            if fit is not None:
                x, y = fit
        ra, dec = c["wcs"].pixel_to_world_values(x, y)
        rows.append({"file": f.name, "date": c["time"].iso[:10], "year": c["time"].decimalyear,
                     "ra": float(ra), "dec": float(dec), "snr": snr,
                     "core_flagged": core_flag, "core_missing": core_nan})

    if len(rows) < 4:
        raise SystemExit(f"Only {len(rows)} usable images (skipped: {skipped[:5]}). Need at least 4.")

    years = np.array([r["year"] for r in rows])
    star_ra, star_dec = ra, dec  # the requested position (None for built-in targets)
    ra = np.array([r["ra"] for r in rows])
    dec = np.array([r["dec"] for r in rows])
    t0, ra0, dec0 = years.mean(), ra.mean(), dec.mean()

    # Offsets on the sky in arcseconds (east = +x, north = +y)
    x_as = (ra - ra0) * np.cos(np.radians(dec0)) * 3600
    y_as = (dec - dec0) * 3600
    dt = years - t0

    # 1. Group images into epochs (one per six-month sky pass) and reject outliers
    #    *within* each epoch. (Clipping against one global line can throw away a
    #    whole epoch, which is exactly the epoch that carries the motion signal.)
    labels = group_epochs(years)
    keep = np.ones(len(years), bool)
    for e in np.unique(labels):
        m = labels == e
        if m.sum() < 3:
            continue
        for v in (x_as, y_as):
            med = np.median(v[m])
            mad = 1.4826 * np.median(np.abs(v[m] - med))
            bad = m & (np.abs(v - med) > 3 * max(mad, 0.3))
            keep &= ~bad

    # 2. Average each epoch; its uncertainty is the scatter / sqrt(n), with a
    #    0.3" floor for systematics (SPHEREx pixels are ~6", undersampled PSF).
    epochs = []
    for e in np.unique(labels):
        m = (labels == e) & keep
        n = int(m.sum())
        if n == 0:
            continue
        ex = x_as[m].std(ddof=1) / np.sqrt(n) if n > 1 else 1.0
        ey = y_as[m].std(ddof=1) / np.sqrt(n) if n > 1 else 1.0
        epochs.append({
            "epoch": len(epochs) + 1,
            "date": Time(years[m].mean(), format="decimalyear").iso[:10],
            "t": float(dt[m].mean()),
            "n_images": n,
            "ra": float(ra[m].mean()), "dec": float(dec[m].mean()),
            "x_arcsec": float(x_as[m].mean()), "y_arcsec": float(y_as[m].mean()),
            "x_err": float(max(ex, 0.3)), "y_err": float(max(ey, 0.3)),
        })
    if len(epochs) < 2:
        raise SystemExit("Need at least 2 epochs (sky passes) with usable images to measure motion.")

    # 3. Weighted straight-line fit through the epoch averages
    def wfit(t, v, err):
        w = 1.0 / np.asarray(err) ** 2
        A = np.vstack([t, np.ones_like(t)]).T
        cov = np.linalg.inv(A.T @ (A * w[:, None]))
        p = cov @ (A.T @ (w * v))
        return float(p[0]), float(np.sqrt(cov[0, 0]))

    te = np.array([e["t"] for e in epochs])
    pmra, pmra_err = wfit(te, np.array([e["x_arcsec"] for e in epochs]), [e["x_err"] for e in epochs])
    pmdec, pmdec_err = wfit(te, np.array([e["y_arcsec"] for e in epochs]), [e["y_err"] for e in epochs])
    total = float(np.hypot(pmra, pmdec))
    total_err = float(np.hypot(pmra * pmra_err, pmdec * pmdec_err) / total) if total > 0 else float("nan")

    # 4. Reference: built-in value, or a live Gaia DR3 lookup for any other star
    moving = total > 3 * total_err and total > 1.0   # clearly moving (> 1"/yr and > 3 sigma)
    reference, gaia_note = None, ""
    if not any_star:
        p = PRESETS[key]
        reference = {"pmra": p["pmra"] / 1000, "pmdec": p["pmdec"] / 1000, "source": REFERENCE_SOURCE[key]}
    else:
        try:
            g = gaia_lookup(float(ra0), float(dec0), float(t0))
            if g is None:
                gaia_note = "no Gaia DR3 star at this position"
            elif not g["has_pm"]:
                gaia_note = f"Gaia DR3 {g['source_id']} has no proper motion measured"
            else:
                reference = {"pmra": g["pmra"], "pmdec": g["pmdec"],
                             "source": f"Gaia DR3 {g['source_id']} (G={g['gmag']}, via {g['service']})"}
        except Exception as exc:
            gaia_note = f"Gaia archive not reachable ({str(exc)[:160]})"

    if reference is not None:
        ref_pmra, ref_pmdec = reference["pmra"], reference["pmdec"]
        ref_total = float(np.hypot(ref_pmra, ref_pmdec))
        diff = total - ref_total
        chi2 = ((pmra - ref_pmra) / pmra_err) ** 2 + ((pmdec - ref_pmdec) / pmdec_err) ** 2
        sigma = float(np.sqrt(chi2))
        verdict = "consistent" if sigma < 3 else "inconsistent"
        if len(epochs) == 2:
            verdict += " (only 2 epochs: weak test)"
        ref_out = {"pmra": round(ref_pmra, 3), "pmdec": round(ref_pmdec, 3), "total": round(ref_total, 3),
                   "source": reference["source"]}
    else:
        ref_total = diff = sigma = None
        ref_out = None
        if gaia_note.startswith("Gaia archive not reachable"):
            verdict = "no reference (Gaia offline)"
        elif moving:
            verdict = "DISCOVERY CANDIDATE: moving, but no Gaia motion known"
        else:
            verdict = "not moving measurably; no Gaia reference"

    result = {
        "target": key or folder.name,
        "star": {"ra": round(float(ra0), 6), "dec": round(float(dec0), 6)} if any_star else None,
        "band": band.upper(),
        "method": method,
        # --- API contract fields ---
        "measured_arcsec_yr": round(total, 3),
        "gaia_arcsec_yr": None if ref_total is None else round(ref_total, 3),
        "positions": [{"date": r["date"], "ra": r["ra"], "dec": r["dec"]} for r, k in zip(rows, keep) if k],
        # --- details ---
        "measured": {"pmra": round(pmra, 3), "pmra_err": round(pmra_err, 3),
                     "pmdec": round(pmdec, 3), "pmdec_err": round(pmdec_err, 3),
                     "total": round(total, 3), "total_err": round(total_err, 3)},
        "reference": ref_out,
        "reference_note": gaia_note,
        "moving": bool(moving),
        "difference_arcsec_yr": None if diff is None else round(diff, 3),
        "difference_sigma": None if sigma is None else round(sigma, 2),
        "difference_note": "sigma compares both components (RA and Dec) with the reference",
        "verdict": verdict,
        "n_used": int(keep.sum()),
        "n_rejected_outliers": int((~keep).sum()),
        "n_epochs": len(epochs),
        "n_skipped": len(skipped),
        "skipped": [{"file": f, "reason": why} for f, why in skipped],
        "n_core_flagged": int(sum(1 for r in rows if r["core_flagged"] > 0)),
        "n_core_missing": int(sum(1 for r in rows if r["core_missing"] > 0)),
        "epochs": epochs,
        "baseline_years": round(float(years.max() - years.min()), 3),
        "note": "Parallax is not modelled; it adds a small yearly wobble (0.55\" for Barnard's Star).",
    }

    suffix = f"_ra{float(star_ra):.4f}_dec{float(star_dec):+.4f}" if any_star else ""
    out_json = folder / f"motion_{band.upper()}{suffix}.json"
    out_json.write_text(json.dumps(result, indent=2))
    if make_plot:
        plot_motion(result, x_as, y_as, dt, keep, folder / f"motion_{band.upper()}{suffix}.png")
    return result


def plot_motion(res, x, y, dt, keep, out_png: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    m = res["measured"]
    r = res["reference"] or {"pmra": np.nan, "pmdec": np.nan, "total": np.nan}
    tt = np.linspace(dt.min(), dt.max(), 50)

    ax = axes[0]
    sc = ax.scatter(x[keep], y[keep], c=dt[keep], cmap="viridis", s=18)
    ax.scatter(x[~keep], y[~keep], marker="x", color="red", s=20, label="rejected")
    ax.plot(m["pmra"] * tt, m["pmdec"] * tt, "k-", lw=1.5, label="SPHEREx fit")
    ax.plot(r["pmra"] * tt, r["pmdec"] * tt, "--", color="orange", lw=1.5, label="reference")
    ax.invert_xaxis()  # east to the left, as on the sky
    ax.set_xlabel("East offset (arcsec)")
    ax.set_ylabel("North offset (arcsec)")
    ax.set_title("Track on the sky")
    ax.legend(fontsize=7)
    fig.colorbar(sc, ax=ax, label="years from mean")

    for ax, v, slope, ref_slope, lab in ((axes[1], x, m["pmra"], r["pmra"], "East"),
                                         (axes[2], y, m["pmdec"], r["pmdec"], "North")):
        ax.scatter(dt[keep], v[keep], s=14)
        ax.scatter(dt[~keep], v[~keep], marker="x", color="red", s=20)
        off = np.median(v[keep] - slope * dt[keep])
        ax.plot(tt, slope * tt + off, "k-", label=f"SPHEREx {slope:+.2f}\"/yr")
        ax.plot(tt, ref_slope * tt + off, "--", color="orange", label=f"reference {ref_slope:+.2f}\"/yr")
        ax.set_xlabel("years from mean")
        ax.set_ylabel(f"{lab} offset (arcsec)")
        ax.legend(fontsize=7)

    fig.suptitle(f"{res['target']} {res['band']}: measured {m['total']:.2f} ± {m['total_err']:.2f}\"/yr, "
                 f"reference {r['total']:.2f}\"/yr  →  {res['verdict']} ({res['difference_sigma']}σ)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Light curve (for the variable stars RW Aurigae and R Draconis)
# ---------------------------------------------------------------------------
def light_curve(target: str = None, band: str = "D1", data_dir="data", r_ap=2, make_plot=True,
                ra=None, dec=None, folder=None):
    """
    Simple aperture photometry in each cutout.
      - target only: a built-in target (its motion is followed)
      - ra/dec + folder (or + target): any source at a fixed position in that field
    Caveat: each SPHEREx image samples a slightly different wavelength inside the
    detector's range, so part of the scatter is colour, not variability.
    """
    any_pos = ra is not None and dec is not None
    key = resolve_target(target) if target else None
    if folder is None:
        if key is None:
            raise SystemExit("Give a target, or a data folder together with ra/dec.")
        folder = Path(data_dir) / PRESETS[key]["label"]
    folder = Path(folder)
    files = sorted(folder.glob(f"*_{band.upper()}_*.fits"))
    if not files:
        raise SystemExit(f"No {band} cutouts in {folder}.")
    pts = []
    for f in files:
        c = load_cutout(f)
        ra_p, dec_p = (ra, dec) if any_pos else predicted_radec(key, c["time"])
        x0, y0 = c["wcs"].world_to_pixel_values(ra_p, dec_p)
        res = centroid(c["img"], float(x0), float(y0))
        if res is None:
            continue
        x, y, _ = res
        img = c["img"]
        yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
        rr = np.hypot(xx - x, yy - y)
        ap = rr <= r_ap
        ring = (rr > r_ap + 2) & (rr <= r_ap + 6)
        bg = np.nanmedian(img[ring])
        flux = float(np.nansum(img[ap] - bg))
        pts.append({"date": c["time"].iso[:10], "year": c["time"].decimalyear, "flux": flux})

    if not pts:
        raise SystemExit("No usable measurements.")
    pts.sort(key=lambda p: p["year"])
    name = key or folder.name
    result = {"target": name, "source": {"ra": ra, "dec": dec} if any_pos else "tracked target",
              "band": band.upper(), "unit": "MJy/sr summed over aperture (relative)",
              "points": [{"date": p["date"], "flux": round(p["flux"], 4)} for p in pts]}
    suffix = f"_ra{ra:.4f}_dec{dec:+.4f}" if any_pos else ""
    (folder / f"lightcurve_{band.upper()}{suffix}.json").write_text(json.dumps(result, indent=2))

    if make_plot:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot([Time(p["year"], format="decimalyear").datetime for p in pts],
                [p["flux"] for p in pts], "o-", ms=4)
        ax.set_ylabel("flux (relative)")
        ax.set_title(f"{name} {band.upper()} light curve")
        fig.autofmt_xdate()
        fig.tight_layout()
        fig.savefig(folder / f"lightcurve_{band.upper()}{suffix}.png", dpi=110)
        plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# Spectrum ("chemical fingerprint")
# ---------------------------------------------------------------------------
# Approximate centres of absorption features seen in cool stars and brown dwarfs.
FEATURES = [
    (1.15, "H\u2082O"), (1.4, "H\u2082O"), (1.65, "CH\u2084"), (1.9, "H\u2082O"),
    (2.2, "CH\u2084"), (2.3, "CO"), (2.7, "H\u2082O"), (3.3, "CH\u2084"),
    (4.3, "CO\u2082"), (4.65, "CO"),
]


def _spectral_wcs(hdul):
    """Wavelength map of a SPHEREx image: alternate WCS 'W' + the WCS-WAVE lookup table."""
    sw = WCS(header=hdul["IMAGE"].header, fobj=hdul, key="W")
    sw.sip = None  # wavelength depends on the filter, not on optical distortion
    return sw


def _aperture_flux(img, var, x, y, r_ap=2.0, r_in=4.0, r_out=7.0):
    """Background-subtracted sum (and error) in a circular aperture, in image units."""
    yy, xx = np.indices(img.shape)
    rr = np.hypot(xx - x, yy - y)
    ap = rr <= r_ap
    ring = (rr > r_in) & (rr <= r_out)
    if ring.sum() < 10 or not np.isfinite(img[ap]).all():
        return None
    bg = np.nanmedian(img[ring])
    s = float(np.nansum(img[ap] - bg))
    e = float(np.sqrt(np.nansum(var[ap]))) if var is not None else float("nan")
    return s, e


def get_spectrum(target=None, ra=None, dec=None, data_dir="data", folder=None,
                 bands=("D1", "D2", "D3", "D4", "D5", "D6"), make_plot=True):
    """
    Build a 0.75-5 micron spectrum of one source from all downloaded cutouts.

    Every SPHEREx image sees a different wavelength at every pixel, so each cutout
    that contains the source gives ONE point of the spectrum: (wavelength at the
    source's pixel, flux of the source). Hundreds of cutouts -> a full spectrum.

    - target only: follow the star (proper motion applied per image date).
    - target + ra/dec: spectrum of ANOTHER source at fixed ra/dec in that target's
      cutouts (e.g. a star the user clicked on).
    """
    key = resolve_target(target) if target else None
    if folder is None:
        if key is None:
            raise SystemExit("Give --target (and optionally --ra/--dec), or a data folder.")
        folder = Path(data_dir) / PRESETS[key]["label"]
    folder = Path(folder)
    follow_star = ra is None or dec is None

    points, skipped = [], 0
    for band in bands:
        for f in sorted(folder.glob(f"*_{band.upper()}_*.fits")):
            try:
                with fits.open(f) as hdul:
                    img = hdul["IMAGE"].data.astype(float)
                    if "ZODI" in hdul:
                        img = img - hdul["ZODI"].data.astype(float)
                    var = hdul["VARIANCE"].data.astype(float) if "VARIANCE" in hdul else None
                    cel = WCS(hdul["IMAGE"].header).celestial
                    t = obs_time([hdul["IMAGE"].header, hdul[0].header], f)
                    swcs = _spectral_wcs(hdul)

                    # Where is the source in this image?
                    if follow_star:
                        ra_p, dec_p = predicted_radec(key, t)
                    else:
                        ra_p, dec_p = ra, dec
                    x, y = (float(v) for v in cel.world_to_pixel_values(ra_p, dec_p))
                    # Re-centre on the star if it is clearly detected close to the prediction
                    c = centroid(img, x, y, search=2)
                    if c is not None and c[2] > 10 and np.hypot(c[0] - x, c[1] - y) < 1.5:
                        x, y = c[0], c[1]

                    res = _aperture_flux(img, var, x, y)
                    if res is None:
                        skipped += 1
                        continue
                    wl, bw = swcs.pixel_to_world(x, y)
                    omega_sr = proj_plane_pixel_area(cel) * (np.pi / 180) ** 2
                    to_mjy = omega_sr * 1e9  # MJy/sr * sr = MJy; 1 MJy = 1e9 mJy
                    points.append({
                        "wavelength_um": float(u.Quantity(wl).to_value(u.um)),
                        "bandwidth_um": float(u.Quantity(bw).to_value(u.um)),
                        "flux_mJy": res[0] * to_mjy,
                        "err_mJy": res[1] * to_mjy,
                        "band": band.upper(),
                        "date": t.iso[:10],
                    })
            except Exception as exc:  # one bad file should not kill the spectrum
                skipped += 1
                print(f"    skipped {f.name}: {type(exc).__name__}: {str(exc)[:80]}")

    if not points:
        raise SystemExit(f"No spectrum points found in {folder}. Download some bands first.")
    points.sort(key=lambda p: p["wavelength_um"])

    # Bin into 102 log-spaced channels (roughly SPHEREx's spectral resolution)
    wl = np.array([p["wavelength_um"] for p in points])
    fl = np.array([p["flux_mJy"] for p in points])
    er = np.array([p["err_mJy"] for p in points])
    edges = np.logspace(np.log10(0.74), np.log10(5.02), 103)
    idx = np.digitize(wl, edges) - 1
    b_wl, b_fl, b_er, b_n = [], [], [], []
    for i in np.unique(idx):
        m = idx == i
        good = m & np.isfinite(er) & (er > 0)
        if good.sum() >= 1:
            w = 1 / er[good] ** 2
            b_fl.append(float(np.sum(w * fl[good]) / w.sum()))
            b_er.append(float(1 / np.sqrt(w.sum())))
        else:
            b_fl.append(float(np.median(fl[m])))
            b_er.append(float("nan"))
        b_wl.append(float(np.mean(wl[m])))
        b_n.append(int(m.sum()))

    label = key or folder.name
    result = {
        "target": label,
        "source": "tracked target" if follow_star else {"ra": ra, "dec": dec},
        "unit": "mJy (2-pixel-radius aperture, not aperture-corrected: shape is reliable, "
                "absolute level is approximate)",
        # --- API contract fields (binned spectrum) ---
        "wavelength_um": [round(v, 4) for v in b_wl],
        "flux": [round(v, 5) for v in b_fl],
        # --- details ---
        "flux_err": [round(v, 5) if np.isfinite(v) else None for v in b_er],
        "n_per_bin": b_n,
        "points": points,
        "n_points": len(points),
        "n_skipped": skipped,
        "bands_used": sorted({p["band"] for p in points}),
        "features": [{"wavelength_um": w, "molecule": m} for w, m in FEATURES],
        "note": "Each point comes from a different image (and date); for variable stars, "
                "brightness changes mix with the spectrum.",
    }
    suffix = "" if follow_star else f"_ra{ra:.4f}_dec{dec:+.4f}"
    (folder / f"spectrum{suffix}.json").write_text(json.dumps(result, indent=2))
    if make_plot:
        plot_spectrum(result, folder / f"spectrum{suffix}.png")
    return result


def plot_spectrum(res, out_png: Path):
    """Plot all image points, the binned spectrum, and molecule markers."""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors = {"D1": "#4c72b0", "D2": "#55a868", "D3": "#c44e52",
              "D4": "#8172b2", "D5": "#ccb974", "D6": "#64b5cd"}
    for band in res["bands_used"]:
        pts = [p for p in res["points"] if p["band"] == band]
        ax.scatter([p["wavelength_um"] for p in pts], [p["flux_mJy"] for p in pts],
                   s=8, alpha=0.35, color=colors.get(band, "grey"), label=f"{band} images")
    ax.plot(res["wavelength_um"], res["flux"], "k-", lw=1.4, label="binned spectrum")
    fl = np.asarray(res["flux"], dtype=float)
    top = float(np.nanmax(fl)) if fl.size else 1.0
    low = min(0.0, float(np.nanmin(fl))) if fl.size else 0.0
    ax.set_ylim(low - 0.05 * top, top * 1.35)  # scale to the spectrum, not to stray single points
    for f in res["features"]:
        ax.axvline(f["wavelength_um"], color="grey", ls=":", lw=0.8)
        ax.text(f["wavelength_um"], top * 1.3, f["molecule"], rotation=90, fontsize=7,
                ha="center", va="top", color="dimgrey")
    ax.set_xlabel("wavelength (\u00b5m)")
    ax.set_ylabel("flux (mJy, aperture)")
    ax.set_xlim(0.7, 5.1)
    ax.set_title(f"SPHEREx spectrum: {res['target']}  ({res['n_points']} image points, "
                 f"bands {', '.join(res['bands_used'])})", fontsize=10)
    ax.legend(fontsize=7, ncol=4, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="SPHEREx science checks")
    ap.add_argument("what", choices=["motion", "lightcurve", "spectrum"])
    ap.add_argument("--target", required=True, help="barnard, epsindib, wise0855, rwaur, rdra")
    ap.add_argument("--ra", type=float, help="spectrum: another source's RA (deg) in the target's cutouts")
    ap.add_argument("--dec", type=float, help="spectrum: another source's Dec (deg)")
    ap.add_argument("--bands", default="D1,D2,D3,D4,D5,D6", help="spectrum: which detectors to use")
    ap.add_argument("--band", default="D1")
    ap.add_argument("--data", default="data")
    ap.add_argument("--min-snr", type=float, default=10.0)
    ap.add_argument("--verbose", action="store_true", help="List skipped images and why")
    ap.add_argument("--method", choices=["psf", "centroid"], default="psf",
                    help="motion: psf = Gaussian fit ignoring saturated pixels (default); centroid = simple")
    ap.add_argument("--exclude-flagged", action="store_true",
                    help="motion: drop images whose star core has flagged or missing pixels (e.g. saturation)")
    args = ap.parse_args()

    if args.what == "motion":
        r = measure_motion(args.target, args.band, args.data, args.min_snr,
                           exclude_flagged=args.exclude_flagged, method=args.method,
                           ra=args.ra, dec=args.dec)
        m, ref = r["measured"], r["reference"]
        print(f"\n{r['target']} ({r['band']}, {r['method']}): {r['n_used']} images used, "
              f"{r['n_rejected_outliers']} outliers rejected, {r['n_skipped']} skipped, "
              f"baseline {r['baseline_years']} yr")
        for e in r["epochs"]:
            print(f"  epoch {e['epoch']}: {e['date']}  n={e['n_images']:2d}  "
                  f"offset E={e['x_arcsec']:+6.2f}±{e['x_err']:.2f}\"  N={e['y_arcsec']:+6.2f}±{e['y_err']:.2f}\"")
        print(f"\n  SPHEREx   : pmRA {m['pmra']:+.2f} ± {m['pmra_err']:.2f}  pmDec {m['pmdec']:+.2f} ± "
              f"{m['pmdec_err']:.2f}  total {m['total']:.2f} ± {m['total_err']:.2f} \"/yr")
        if ref:
            print(f"  Reference : pmRA {ref['pmra']:+.2f}  pmDec {ref['pmdec']:+.2f}  "
                  f"total {ref['total']:.2f} \"/yr   [{ref['source']}]")
            print(f"  Speed difference {r['difference_arcsec_yr']:+.2f} \"/yr; "
                  f"vector difference {r['difference_sigma']}σ  →  {r['verdict'].upper()}")
        else:
            print(f"  Reference : none ({r['reference_note']})  →  {r['verdict'].upper()}")
        n_tot = r["n_used"] + r["n_rejected_outliers"]
        if r["n_core_missing"]:
            print(f"  Data quality: star core has MISSING pixels in {r['n_core_missing']}/{n_tot} images "
                  f"-> saturated; the psf method fits around them")
        elif r["n_core_flagged"]:
            print(f"  Data quality: no missing pixels (not saturated); core carries flags in "
                  f"{r['n_core_flagged']}/{n_tot} images (flags also mark ordinary sources)")
        if args.verbose and r["skipped"]:
            print("\n  Skipped images:")
            for sk in r["skipped"]:
                print(f"    {sk['file']}: {sk['reason']}")
        print(f"\n  Saved motion_{r['band']}*.json and .png in the target's data folder")
    elif args.what == "spectrum":
        r = get_spectrum(args.target, args.ra, args.dec, args.data,
                         bands=[b.strip() for b in args.bands.split(",")])
        print(f"\n{r['target']}: {r['n_points']} spectrum points from bands {', '.join(r['bands_used'])} "
              f"({r['n_skipped']} skipped)")
        print(f"  wavelength range {r['wavelength_um'][0]:.2f} - {r['wavelength_um'][-1]:.2f} um, "
              f"{len(r['wavelength_um'])} channels")
        peak = int(np.nanargmax(r["flux"]))
        print(f"  brightest at {r['wavelength_um'][peak]:.2f} um ({r['flux'][peak]:.2f} mJy)")
        print("  Saved spectrum.json and spectrum.png in the target's data folder")
    else:
        r = light_curve(args.target, args.band, args.data)
        print(f"{len(r['points'])} points saved to lightcurve_{r['band']}.json / .png")


if __name__ == "__main__":
    main()
