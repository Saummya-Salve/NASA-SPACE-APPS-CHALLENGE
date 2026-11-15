"""
Stage 1: fetch SPHEREx cutouts for a sky position and save quick-look PNGs.

Usage examples (run from your project folder, with the .venv active):

    python fetch_spherex.py --name "Barnard's Star"
    python fetch_spherex.py --name "WISE J085510.83-071442.5" --band D6
    python fetch_spherex.py --ra 269.4521 --dec 4.6934 --size 0.1

What it does:
  1. Resolves the target (by name via SIMBAD, or RA/Dec you give it).
     For named targets it applies proper motion, so fast movers like
     Barnard's Star are centred where they are *now*, not in the year 2000.
  2. Asks IRSA which SPHEREx spectral images cover that position (TAP query).
  3. Prints how many images exist per detector (D1-D6) and their dates.
  4. Downloads small cutouts for one detector into data/<target>/
     (files already downloaded are skipped, so re-runs are fast).
  5. Saves a quick-look PNG of each cutout (zodiacal light subtracted,
     same brightness scaling for all frames) so you can flip through them.

Install once:
    pip install astropy astroquery numpy matplotlib
"""

import argparse
import random
import re
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests  # installed with astroquery; uses its own certificate bundle

import astropy.units as u
import matplotlib

matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astroquery.ipac.irsa import Irsa

# Harmless astropy note about the dummy distance used for proper motion.
warnings.filterwarnings("ignore", message=".*distance overridden.*")

# SPHEREx observations started in 2025; propagate proper motion to roughly
# the middle of the mission data so far.
TARGET_EPOCH = Time("2025-11-01")


# ---------------------------------------------------------------------------
# 1. Resolve the target
# ---------------------------------------------------------------------------
# Built-in test targets, so the demo never depends on an online name lookup.
# ra/dec in degrees at epoch `ref` (decimal year); pm in mas/yr.
PRESETS = {
    "barnard": dict(label="Barnards_Star", ra=269.452076, dec=4.693364,
                    pmra=-801.6, pmdec=10362.4, ref=2000.0),
    "epsindib": dict(label="Epsilon_Indi_B", ra=331.04383, dec=-56.78269,
                     pmra=3966.7, pmdec=-2536.2, ref=2000.0),
    "wise0855": dict(label="WISE_0855", ra=133.795129, dec=-7.245147,
                     pmra=-8100.0, pmdec=700.0, ref=2010.5),
    "rwaur": dict(label="RW_Aurigae", ra=76.95642, dec=30.40142,
                  pmra=0.0, pmdec=0.0, ref=2000.0),
    "rdra": dict(label="R_Draconis", ra=248.16742, dec=66.75500,
                 pmra=0.0, pmdec=0.0, ref=2000.0),
}
ALIASES = {
    "barnardsstar": "barnard", "barnard": "barnard", "barnardstar": "barnard",
    "epsilonindib": "epsindib", "epsindib": "epsindib", "epsilonindi": "epsindib",
    "wise0855": "wise0855", "wisej085510830714425": "wise0855",
    "rwaurigae": "rwaur", "rwaur": "rwaur",
    "rdraconis": "rdra", "rdra": "rdra",
}


def move_to_now(ra, dec, pmra, pmdec, ref_year) -> SkyCoord:
    """Shift a position along its proper motion from ref_year to TARGET_EPOCH."""
    if pmra == 0 and pmdec == 0:
        return SkyCoord(ra * u.deg, dec * u.deg)
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg,
                 pm_ra_cosdec=pmra * u.mas / u.yr, pm_dec=pmdec * u.mas / u.yr,
                 distance=1 * u.kpc, obstime=Time(ref_year, format="decimalyear"),
                 frame="icrs")
    moved = c.apply_space_motion(new_obstime=TARGET_EPOCH)
    print(f"  proper motion {pmra:.0f}, {pmdec:.0f} mas/yr -> shifted "
          f"{c.separation(moved).to(u.arcsec):.1f} since {ref_year}")
    return SkyCoord(moved.ra, moved.dec)


def resolve_name(name: str):
    """Return (coord, label). Uses built-in presets first, then online lookups."""
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    if key in ALIASES:
        p = PRESETS[ALIASES[key]]
        print("  using built-in position for this test target")
        return move_to_now(p["ra"], p["dec"], p["pmra"], p["pmdec"], p["ref"]), p["label"]

    label = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    try:
        return resolve_simbad(name), label
    except Exception as exc:
        print(f"  SIMBAD lookup failed ({type(exc).__name__}); trying Sesame without proper motion")
    try:
        return SkyCoord.from_name(name), label
    except Exception:
        raise SystemExit(f"Could not resolve '{name}'. Use --ra and --dec instead.")


def resolve_simbad(name: str) -> SkyCoord:
    """Look up a name in SIMBAD and move it along its proper motion to TARGET_EPOCH."""
    from astroquery.simbad import Simbad

    simbad = Simbad()
    simbad.add_votable_fields("pmra", "pmdec")
    result = simbad.query_object(name)
    if result is None or len(result) == 0:
        raise ValueError(f"SIMBAD could not find '{name}'")

    row = result[0]
    cols = {c.lower(): c for c in result.colnames}

    # Newer astroquery returns ra/dec in degrees; older returns sexagesimal strings.
    ra_val, dec_val = row[cols["ra"]], row[cols["dec"]]
    if isinstance(ra_val, str):
        c0 = SkyCoord(ra_val, dec_val, unit=(u.hourangle, u.deg))
        ra_deg, dec_deg = c0.ra.deg, c0.dec.deg
    else:
        ra_deg, dec_deg = float(ra_val), float(dec_val)

    pmra = row[cols["pmra"]] if "pmra" in cols else None
    pmdec = row[cols["pmdec"]] if "pmdec" in cols else None
    has_pm = pmra is not None and pmdec is not None and not (
        np.ma.is_masked(pmra) or np.ma.is_masked(pmdec)
    )

    if not has_pm:
        print(f"  (no proper motion in SIMBAD for {name}; using catalogue position)")
        return SkyCoord(ra_deg * u.deg, dec_deg * u.deg)

    c2000 = SkyCoord(
        ra=ra_deg * u.deg,
        dec=dec_deg * u.deg,
        pm_ra_cosdec=float(pmra) * u.mas / u.yr,
        pm_dec=float(pmdec) * u.mas / u.yr,
        obstime=Time("J2000"),
        frame="icrs",
    )
    # apply_space_motion needs a distance; a large dummy value is fine for
    # positions (it only affects tiny parallax/perspective terms here).
    c2000 = SkyCoord(
        ra=c2000.ra, dec=c2000.dec,
        pm_ra_cosdec=c2000.pm_ra_cosdec, pm_dec=c2000.pm_dec,
        distance=1 * u.kpc, obstime=c2000.obstime, frame="icrs",
    )
    moved = c2000.apply_space_motion(new_obstime=TARGET_EPOCH)
    shift = c2000.separation(moved).to(u.arcsec)
    print(f"  proper motion {float(pmra):.0f}, {float(pmdec):.0f} mas/yr "
          f"-> shifted {shift:.1f} since J2000")
    return SkyCoord(moved.ra, moved.dec)


# ---------------------------------------------------------------------------
# 2. Ask IRSA which SPHEREx images cover the position
# ---------------------------------------------------------------------------
def query_spherex(coord: SkyCoord, size_deg: float):
    """Return a table of every SPHEREx spectral image covering coord, with cutout URLs."""
    ra, dec = coord.ra.deg, coord.dec.deg
    query = f"""
    SELECT
        'https://irsa.ipac.caltech.edu/' || a.uri ||
            '?center={ra},{dec}d&size={size_deg}' AS cutout_url,
        p.energy_bandpassname AS band,
        p.time_bounds_lower AS mjd
    FROM spherex.artifact a
    JOIN spherex.plane p ON a.planeid = p.planeid
    WHERE 1 = CONTAINS(POINT('ICRS', {ra}, {dec}), p.poly)
    ORDER BY p.time_bounds_lower
    """
    # IRSA is often overloaded (502/503). Retry the normal (synchronous) search a few
    # times with growing pauses, then try IRSA's asynchronous job queue, which is a
    # separate route into the same archive.
    last = None
    for attempt in range(3):
        try:
            return Irsa.query_tap(query).to_table()
        except Exception as exc:
            last = exc
            wait = 10 * (attempt + 1)
            print(f"  IRSA search failed ({str(exc)[:70]}); retrying in {wait} s...")
            time.sleep(wait)
    try:
        print("  trying IRSA's asynchronous search queue...")
        return Irsa.query_tap(query, async_job=True).to_table()
    except Exception as exc:
        raise RuntimeError(f"IRSA search failed after retries: {str(last)[:120]}") from exc


# ---------------------------------------------------------------------------
# 3. Download cutouts
# ---------------------------------------------------------------------------
def download(url: str, path: Path, attempts: int = 4) -> bool:
    """Download with retries. IRSA returns 503 / drops connections when busy, so wait and try again."""
    if path.exists() and path.stat().st_size > 0:
        return True
    tmp = path.with_suffix(".part")
    last = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(20, 180)) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"{r.status_code} server busy")
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 16):
                        fh.write(chunk)
            tmp.replace(path)
            return True
        except Exception as exc:
            last = exc
            if tmp.exists():
                tmp.unlink()
            if attempt < attempts:
                wait = 5 * 2 ** (attempt - 1) + random.uniform(0, 3)  # ~5, 10, 20 s
                time.sleep(wait)
    print(f"    ! gave up after {attempts} tries: {str(last)[:120]}")
    return False


# ---------------------------------------------------------------------------
# 4. Quick-look PNGs
# ---------------------------------------------------------------------------
def load_image(path: Path):
    """Return (sky image with zodi removed, wavelength at centre or None)."""
    with fits.open(path) as hdul:
        img = hdul["IMAGE"].data.astype(float)
        if "ZODI" in hdul:
            img = img - hdul["ZODI"].data.astype(float)
    return img


def save_pngs(files, out_dir: Path, title_prefix: str):
    images = [(f, load_image(f["path"])) for f in files]
    if not images:
        return

    # One shared brightness scale for all frames, so changes are real changes.
    all_pixels = np.concatenate([im[np.isfinite(im)].ravel() for _, im in images])
    vmin, vmax = np.percentile(all_pixels, [1, 99.5])

    for f, im in images:
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        ax.imshow(np.arcsinh((im - vmin) / (vmax - vmin) * 10), origin="lower",
                  cmap="gray", vmin=0, vmax=np.arcsinh(10))
        ax.plot(im.shape[1] / 2 - 0.5, im.shape[0] / 2 - 0.5, "+", color="red", ms=14)
        ax.set_title(f"{title_prefix}  {f['band']}  {f['date']}", fontsize=8)
        ax.axis("off")
        png = out_dir / (f["path"].stem + ".png")
        fig.savefig(png, bbox_inches="tight")
        plt.close(fig)
    print(f"  saved {len(images)} PNGs to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fetch SPHEREx cutouts for a position.")
    ap.add_argument("--name", help="Target name, e.g. \"Barnard's Star\"")
    ap.add_argument("--ra", type=float, help="Right Ascension in degrees")
    ap.add_argument("--dec", type=float, help="Declination in degrees")
    ap.add_argument("--size", type=float, default=0.05,
                    help="Cutout size in degrees (default 0.05 = 3 arcmin)")
    ap.add_argument("--band", default="D1",
                    help="Detector to download: D1..D6 (default D1)")
    ap.add_argument("--workers", type=int, default=3,
                    help="Parallel downloads (default 3; use 1-2 if IRSA is very busy)")
    ap.add_argument("--max", type=int, default=40,
                    help="Max cutouts to download (default 40)")
    args = ap.parse_args()

    if args.name:
        print(f"Resolving {args.name} ...")
        coord, label = resolve_name(args.name)
    elif args.ra is not None and args.dec is not None:
        coord = SkyCoord(args.ra * u.deg, args.dec * u.deg)
        label = f"ra{args.ra:.4f}_dec{args.dec:+.4f}"
    else:
        raise SystemExit("Give either --name or both --ra and --dec.")

    print(f"Position: RA {coord.ra.deg:.5f}, Dec {coord.dec.deg:+.5f}")
    print("Querying IRSA for SPHEREx images ...")
    table = query_spherex(coord, args.size)
    if len(table) == 0:
        raise SystemExit("No SPHEREx images cover this position yet.")

    # Summary per detector
    dates = Time(np.asarray(table["mjd"], dtype=float), format="mjd")
    table["date"] = [d[:10] for d in dates.iso]
    print(f"\nFound {len(table)} images:")
    for band in sorted(set(table["band"])):
        rows = table[table["band"] == band]
        print(f"  {band}: {len(rows):3d} images, {rows['date'][0]} -> {rows['date'][-1]}")

    # Download one detector
    want = f"SPHEREx-{args.band.upper()}"
    rows = table[table["band"] == want]
    if len(rows) == 0:
        raise SystemExit(f"No images for {want} here.")
    if len(rows) > args.max:
        # Spread the picks evenly across time rather than taking the first N.
        idx = np.linspace(0, len(rows) - 1, args.max).round().astype(int)
        rows = rows[np.unique(idx)]

    out_dir = Path("data") / label
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {len(rows)} {want} cutouts to {out_dir} ...")

    jobs = []
    for i, row in enumerate(rows, 1):
        path = out_dir / f"{row['date']}_{args.band.upper()}_{i:03d}.fits"
        jobs.append((i, str(row["date"]), str(row["cutout_url"]), path))

    def fetch(job):
        i, date, url, path = job
        ok = download(url, path)
        print(f"  [{i}/{len(jobs)}] {date} {'ok' if ok else 'FAILED'}")
        return job, ok

    files = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for (i, date, url, path), ok in pool.map(fetch, jobs):
            if ok:
                files.append({"path": path, "band": args.band.upper(), "date": date})
    print(f"  {len(files)}/{len(jobs)} cutouts downloaded")
    if len(files) < len(jobs):
        print("  Some downloads failed (IRSA busy). Run the SAME command again later:")
        print("  files you already have are skipped, so only the missing ones are retried.")
    if not files:
        raise SystemExit("No cutouts downloaded - paste the error above into the chat.")

    print("\nMaking quick-look PNGs ...")
    save_pngs(files, out_dir, label)
    print("\nDone. Open the PNGs in date order and flip between the first and last.")


if __name__ == "__main__":
    main()
