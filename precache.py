"""
precache.py - prepare everything the demo needs, so nothing waits on NASA IRSA during judging.

For each test target it:
  * makes sure the cutouts are downloaded (live from IRSA, missing ones only)
  * builds the website frames (movie, blink, difference, deep) for the useful detectors
  * runs the science: motion, light curve, spectrum
and prints a summary table at the end.

Run (project folder, .venv active):
    python precache.py                 # everything (can take 30-60 min if IRSA is slow)
    python precache.py --only barnard  # one target
    python precache.py --no-fetch      # only use files already downloaded (fast, offline)

Safe to run again: finished work is reused, failed downloads are retried.
"""

import argparse
import time
import traceback

import analysis
import pipeline

ALL_BANDS = ("D1", "D2", "D3", "D4", "D5", "D6")

# What the demo needs for each target
PLAN = {
    "barnard":  {"frames": ["D3", "D1"], "motion": ["D3", "D1"]},
    "epsindib": {"frames": ["D6", "D3"], "motion": ["D6", "D3"], "spectrum": True},
    "wise0855": {"frames": ["D6", "D5"], "motion": ["D6"]},
    "rwaur":    {"frames": ["D1"], "lightcurve": ["D1"]},
    "rdra":     {"frames": ["D1"], "lightcurve": ["D1"]},
}


def step(results, target, what, fn):
    t0 = time.time()
    print(f"\n--- {target}: {what} ---")
    try:
        info = fn()
        results.append((target, what, "OK", info, time.time() - t0))
        print(f"    OK  {info}")
    except BaseException as exc:  # includes SystemExit from the analysis helpers
        if isinstance(exc, KeyboardInterrupt):
            raise
        msg = str(exc) or type(exc).__name__
        results.append((target, what, "FAILED", msg[:90], time.time() - t0))
        print(f"    FAILED: {msg}")
        if not isinstance(exc, (SystemExit, FileNotFoundError, ValueError)):
            traceback.print_exc(limit=2)


def main():
    ap = argparse.ArgumentParser(description="Pre-cache all demo targets")
    ap.add_argument("--only", help="one target key: barnard, epsindib, wise0855, rwaur, rdra")
    ap.add_argument("--no-fetch", action="store_true", help="do not contact IRSA; use files on disk")
    args = ap.parse_args()
    fetch = not args.no_fetch

    plan = {args.only: PLAN[args.only]} if args.only else PLAN
    results = []
    for target, todo in plan.items():
        # 1. Data (one IRSA search per target covers all bands; then only missing files are fetched)
        bands = set(todo.get("frames", [])) | set(todo.get("motion", [])) | set(todo.get("lightcurve", []))
        if todo.get("spectrum"):
            bands |= set(ALL_BANDS)
        if fetch:
            step(results, target, f"download {','.join(sorted(bands))}",
                 lambda: "latest image " + str(pipeline.ensure_data(target, bands=tuple(sorted(bands)))[1]))

        # 2. Website frames
        for band in todo.get("frames", []):
            step(results, target, f"frames {band}",
                 lambda b=band: (lambda m: f"{m['n_images_total']} images, {m['n_epochs']} epochs")(
                     pipeline.build_frames(target, band=b, fetch=False, force=True)))

        # 3. Science
        for band in todo.get("motion", []):
            step(results, target, f"motion {band}",
                 lambda b=band: (lambda r: f"{r['measured']['total']:.2f}±{r['measured']['total_err']:.2f}\"/yr "
                                            f"vs {r['reference']['total']:.2f} -> {r['verdict']}")(
                     analysis.measure_motion(target, b, make_plot=True)))
        for band in todo.get("lightcurve", []):
            step(results, target, f"light curve {band}",
                 lambda b=band: f"{len(analysis.light_curve(target, b)['points'])} points")
        if todo.get("spectrum"):
            step(results, target, "spectrum",
                 lambda: (lambda s: f"{s['n_points']} points, bands {','.join(s['bands_used'])}")(
                     analysis.get_spectrum(target)))

    print("\n" + "=" * 100)
    print(f"{'target':10} {'step':22} {'result':7} {'time':>6}  details")
    print("-" * 100)
    for target, what, res, info, dt in results:
        print(f"{target:10} {what:22} {res:7} {dt:5.0f}s  {info}")
    failed = [r for r in results if r[2] != "OK"]
    print("=" * 100)
    if failed:
        print(f"{len(failed)} step(s) failed. If IRSA was busy, just run this script again later.")
    else:
        print("Everything cached. The demo will load instantly.")


if __name__ == "__main__":
    main()
