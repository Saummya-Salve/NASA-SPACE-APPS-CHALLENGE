"""
check_data.py - verify downloaded SPHEREx cutouts.

For every target folder in data/ and every detector band it checks that each file:
  * opens as a valid FITS file (not truncated / corrupted)
  * contains the expected layers: IMAGE, FLAGS, VARIANCE, ZODI, WCS-WAVE (+ PSF)
  * has real image data (not empty, not all missing)
  * actually contains the target (the star falls inside the cutout)
It also finds leftover half-downloads (*.part) and counts epochs (sky passes).

Usage (project folder, .venv active):
    python check_data.py                      # check everything in data/
    python check_data.py --target barnard     # one target
    python check_data.py --compare-irsa       # also ask IRSA how many images exist (slow, 1-3 min/target)
    python check_data.py --move-bad           # move broken files to data/_bad so they get re-downloaded
"""

import argparse
import re
import shutil
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from astropy import log as astropy_log
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS, FITSFixedWarning

from fetch_spherex import ALIASES, PRESETS

warnings.simplefilter("ignore", FITSFixedWarning)
astropy_log.setLevel("WARNING")

REQUIRED = ["IMAGE", "FLAGS", "VARIANCE", "ZODI", "WCS-WAVE"]
BANDS = ["D1", "D2", "D3", "D4", "D5", "D6"]


def predicted_radec(p, t: Time):
    dt = t.decimalyear - p["ref"]
    dec = p["dec"] + p["pmdec"] * dt / 3.6e6
    ra = p["ra"] + p["pmra"] * dt / 3.6e6 / np.cos(np.radians(dec))
    return ra, dec


def file_date(path: Path):
    m = re.search(r"(\d{4}-\d{2}-\d{2})", path.name)
    return Time(m.group(1)) if m else None


def check_file(path: Path, preset):
    """Return (ok, problem_text). problem_text is '' when ok."""
    if path.stat().st_size < 10_000:
        return False, f"tiny file ({path.stat().st_size} bytes) - probably an error page"
    try:
        with fits.open(path) as hdul:
            names = [h.name for h in hdul]
            missing = [n for n in REQUIRED if n not in names]
            if missing:
                return False, f"missing layers: {', '.join(missing)}"
            img = hdul["IMAGE"].data
            if img is None or img.size == 0:
                return False, "IMAGE layer is empty"
            if not np.isfinite(img).any():
                return False, "IMAGE is all missing values"
            frac_bad = 1 - np.isfinite(img).mean()
            if preset is not None:
                t = file_date(path)
                if t is not None:
                    wcs = WCS(hdul["IMAGE"].header).celestial
                    ra, dec = predicted_radec(preset, t)
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    ny, nx = img.shape
                    if not (0 <= x < nx and 0 <= y < ny):
                        return False, "target is NOT inside this cutout"
                    if not (3 <= x < nx - 3 and 3 <= y < ny - 3):
                        return True, "warning: target very close to the cutout edge"
            if "PSF" not in names and "EPSF" not in names:
                return True, "warning: no PSF layer"
            if frac_bad > 0.5:
                return True, f"warning: {frac_bad:.0%} of pixels missing"
    except Exception as exc:
        return False, f"cannot open: {type(exc).__name__}: {str(exc)[:60]}"
    return True, ""


def count_epochs(dates, gap_days=60):
    if not dates:
        return 0
    yrs = sorted(d.decimalyear for d in dates)
    return 1 + sum(1 for a, b in zip(yrs[:-1], yrs[1:]) if (b - a) * 365.25 > gap_days)


def irsa_counts(key):
    """Ask IRSA how many images exist per band for this target (slow)."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from fetch_spherex import move_to_now, query_spherex
    p = PRESETS[key]
    coord = move_to_now(p["ra"], p["dec"], p["pmra"], p["pmdec"], p["ref"])
    table = query_spherex(SkyCoord(coord.ra, coord.dec), 0.05)
    counts = defaultdict(int)
    for b in table["band"]:
        counts[str(b).replace("SPHEREx-", "")] += 1
    return counts


def main():
    ap = argparse.ArgumentParser(description="Verify downloaded SPHEREx cutouts")
    ap.add_argument("--data", default="data")
    ap.add_argument("--target", help="only check this target (barnard, epsindib, wise0855, rwaur, rdra)")
    ap.add_argument("--compare-irsa", action="store_true", help="also show how many images IRSA has")
    ap.add_argument("--move-bad", action="store_true", help="move broken files to data/_bad")
    args = ap.parse_args()

    data = Path(args.data)
    label_to_key = {p["label"]: k for k, p in PRESETS.items()}
    folders = sorted(d for d in data.iterdir() if d.is_dir() and not d.name.startswith("_"))
    if args.target:
        key = ALIASES.get(re.sub(r"[^a-z0-9]", "", args.target.lower()))
        if key is None:
            raise SystemExit(f"Unknown target {args.target}")
        folders = [data / PRESETS[key]["label"]]

    all_bad = []
    for folder in folders:
        if not folder.exists():
            print(f"\n{folder.name}: folder does not exist")
            continue
        key = label_to_key.get(folder.name)
        preset = PRESETS.get(key) if key else None
        print(f"\n=== {folder.name} ===")

        parts = list(folder.glob("*.part"))
        if parts:
            print(f"  ! {len(parts)} half-finished downloads (*.part) - safe to delete")

        irsa = irsa_counts(key) if (args.compare_irsa and key) else {}
        header = f"  {'band':4} {'files':>5} {'ok':>4} {'bad':>4} {'warn':>4} {'epochs':>6}  date range"
        if irsa:
            header += "   (IRSA has)"
        print(header)

        for band in BANDS:
            files = sorted(folder.glob(f"*_{band}_*.fits"))
            if not files and not irsa.get(band):
                continue
            ok = bad = warn = 0
            dates, problems = [], []
            for f in files:
                good, msg = check_file(f, preset)
                if good:
                    ok += 1
                    d = file_date(f)
                    if d is not None:
                        dates.append(d)
                    if msg:
                        warn += 1
                        problems.append((f, msg))
                else:
                    bad += 1
                    problems.append((f, msg))
                    all_bad.append(f)
            rng = f"{min(dates).iso[:10]} -> {max(dates).iso[:10]}" if dates else "-"
            line = f"  {band:4} {len(files):5d} {ok:4d} {bad:4d} {warn:4d} {count_epochs(dates):6d}  {rng}"
            if irsa:
                line += f"   ({irsa.get(band, 0)})"
            print(line)
            for f, msg in problems:
                print(f"       {'BAD ' if f in all_bad else 'warn'} {f.name}: {msg}")

    print()
    if all_bad:
        print(f"{len(all_bad)} broken file(s) found.")
        if args.move_bad:
            bad_dir = data / "_bad"
            bad_dir.mkdir(exist_ok=True)
            for f in all_bad:
                shutil.move(str(f), bad_dir / f"{f.parent.name}__{f.name}")
            print(f"Moved them to {bad_dir}. Re-run the same fetch_spherex.py commands to download them again.")
        else:
            print("Run with --move-bad to move them aside, then re-run fetch_spherex.py to re-download.")
    else:
        print("All files OK.")
    print("Tip: you need at least 2 epochs per band to measure motion; 3+ is better.")


if __name__ == "__main__":
    main()
