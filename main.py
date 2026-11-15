"""
main.py - the web server. Serves the website (static/) and the API.

Run:   uvicorn main:app --reload
Open:  http://127.0.0.1:8000        API docs: http://127.0.0.1:8000/docs

Endpoints
  GET /api/targets                         built-in test targets
  GET /api/frames?target=barnard&band=D1   aligned frames, blink, diff, deep (live from IRSA, cached)
  GET /api/frames?ra=..&dec=..&band=D2     same for any position
  GET /api/motion?target=epsindib&band=D6  proper motion measured vs reference
  GET /api/spectrum?target=epsindib        0.75-5 um spectrum of the target
  GET /api/spectrum?target=epsindib&ra=..&dec=..   spectrum of another source in that field
  GET /api/lightcurve?target=rdra          brightness vs time (variable stars)
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

import analysis
import movers
import pipeline
from fetch_spherex import PRESETS

app = FastAPI(title="SPHEREx Sky Time-Lapse Explorer")

ALL_BANDS = ("D1", "D2", "D3", "D4", "D5", "D6")

# Best detector per test target (where the object is bright but not too saturated)
BEST_BAND = {"barnard": "D3", "epsindib": "D6", "wise0855": "D6", "rwaur": "D1", "rdra": "D1"}
INFO = {
    "barnard": ("Barnard's Star", "A red dwarf 6 light-years away and one of the fastest-moving stars "
                "in the sky (about 10 arcseconds per year). Watch it march north between frames."),
    "epsindib": ("Epsilon Indi B", "A pair of cold brown dwarfs orbiting the star Epsilon Indi. Its "
                 "spectrum shows deep water and methane absorption: an atmosphere colder than a star's."),
    "wise0855": ("WISE 0855", "The coldest known brown dwarf, about as cold as the North Pole. "
                 "It is almost invisible at short wavelengths; look in D5/D6 and in the Deep view."),
    "rwaur": ("RW Aurigae", "A young star still surrounded by its planet-forming disk. Its brightness "
              "changes as dust passes in front of it: compare the frames, not the position."),
    "rdra": ("R Draconis", "A pulsating giant star whose brightness rises and falls over about 8 months. "
             "Watch it brighten and fade between frames."),
}
KIND = {"barnard": "motion", "epsindib": "motion", "wise0855": "motion", "rwaur": "variable", "rdra": "variable"}


def _fail(exc):
    """Turn our errors (including SystemExit from the analysis CLI helpers) into clean HTTP errors."""
    msg = str(exc) or type(exc).__name__
    code = 404 if isinstance(exc, (FileNotFoundError,)) else 422
    raise HTTPException(status_code=code, detail=msg)


def _key(target):
    if not target:
        return None
    try:
        return pipeline.resolve(target)[0]
    except ValueError as exc:
        _fail(exc)


def _cached_views(label):
    """Which detector bands / fields of view of this target are already built or downloaded."""
    frames, data_bands = [], set()
    for mpath in pipeline.FRAMES_DIR.glob(f"{label}_*/manifest.json"):
        try:
            m = json.loads(mpath.read_text())
            if m.get("frames") and all((Path("static") / f["url"].lstrip("/")).is_file() for f in m["frames"]):
                frames.append({"band": m["band"], "size_deg": m.get("size_deg", 0.05),
                               "epochs": len(m["frames"])})
        except (OSError, ValueError, KeyError, TypeError):
            continue
    folder = Path("data") / label
    for b in ALL_BANDS:
        if folder.is_dir() and any(folder.glob(f"*_{b}_*.fits")):
            data_bands.add(b)
    return sorted(frames, key=lambda v: (v["band"], v["size_deg"])), sorted(data_bands)


@app.get("/api/targets")
def targets():
    """
    Built-in targets. For each one:
      cached_views: band + field of view combinations whose frames are ready (load instantly)
      data_bands:   detector bands whose SPHEREx images are already downloaded
    Anything else is fetched live from IRSA on request (slower).
    """
    out = []
    for key, p in PRESETS.items():
        name, text = INFO.get(key, (p["label"], ""))
        views, bands = _cached_views(p["label"])
        out.append({"key": key, "name": name, "description": text, "kind": KIND.get(key, "motion"),
                    "best_band": BEST_BAND.get(key, "D1"), "cached_views": views, "data_bands": bands})
    return out


@app.get("/api/frames")
def frames(target: str = None, ra: float = None, dec: float = None,
           band: str = "D1", size: float = Query(0.05, ge=0.02, le=0.2), fetch: bool = True):
    try:
        m = pipeline.build_frames(target, ra, dec, band, size, fetch=fetch)
    except (Exception, SystemExit) as exc:
        _fail(exc)
    key = m["target"]["key"]
    if key:
        m["target"]["name"], m["target"]["description"] = INFO.get(key, (key, ""))
        m["target"]["kind"] = KIND.get(key)
    return m


@app.get("/api/motion")
def motion(target: str = None, band: str = None, ra: float = None, dec: float = None,
           field: str = None, fetch: bool = True):
    """
    Built-in target:   /api/motion?target=barnard&band=D3
    Any star clicked:  /api/motion?field=Barnards_Star&ra=..&dec=..&band=D3
                       (field = the data folder of the view; reference = live Gaia DR3 lookup)
    """
    key = _key(target) if target else None
    band = (band or BEST_BAND.get(key, "D1")).upper()
    try:
        if ra is not None and dec is not None:
            if field:
                folder = Path("data") / Path(field).name   # .name: never leave data/
            elif key:
                folder = Path("data") / PRESETS[key]["label"]
            else:
                raise ValueError("Give field (the data folder of the current view) with ra/dec.")
            if not folder.is_dir():
                raise FileNotFoundError(f"No data folder {folder.name}")
            return analysis.measure_motion(None, band, min_snr=5.0, make_plot=False,
                                           ra=ra, dec=dec, folder=folder)
        if key is None:
            raise ValueError("Give a target, or field + ra + dec.")
        if fetch:
            pipeline.ensure_data(target, bands=(band,))
        return analysis.measure_motion(target, band, make_plot=True)
    except (Exception, SystemExit) as exc:
        _fail(exc)


@app.get("/api/spectrum")
def spectrum(target: str = None, ra: float = None, dec: float = None, fetch: bool = True,
             cached_only: bool = False):
    try:
        key, label, _, _ = pipeline.resolve(target, ra, dec) if target else pipeline.resolve(None, ra, dec)
        folder = pipeline.data_folder(label, pipeline.DEFAULT_SIZE)
        # Cached result for the target itself, if newer than every image
        cached = folder / "spectrum.json"
        if target and ra is None and cached.exists():
            newest = max((f.stat().st_mtime for f in folder.glob("*.fits")), default=0)
            if cached.stat().st_mtime > newest or cached_only:
                return json.loads(cached.read_text())
        if cached_only:
            raise FileNotFoundError("No saved spectrum for this target yet.")
        if fetch:
            pipeline.ensure_data(target, ra if not target else None, dec if not target else None,
                                 bands=ALL_BANDS)
        if target:
            return analysis.get_spectrum(target, ra, dec)
        return analysis.get_spectrum(None, ra, dec, folder=folder)
    except (Exception, SystemExit) as exc:
        _fail(exc)


@app.get("/api/lightcurve")
def lightcurve(target: str = None, ra: float = None, dec: float = None, band: str = "D1",
               fetch: bool = True):
    """Built-in target: ?target=rdra   Any source: ?ra=..&dec=.. (optionally with target for its field)."""
    b = band.upper()
    try:
        if target and (ra is None or dec is None):
            _key(target)
            if fetch:
                pipeline.ensure_data(target, bands=(b,))
            return analysis.light_curve(target, b)
        if ra is None or dec is None:
            raise ValueError("Give a target, or ra and dec.")
        if target:
            if fetch:
                pipeline.ensure_data(target, bands=(b,))
            return analysis.light_curve(target, b, ra=ra, dec=dec)
        folder, _ = pipeline.ensure_data(None, ra, dec, bands=(b,)) if fetch else (
            pipeline.data_folder(pipeline.resolve(None, ra, dec)[1], pipeline.DEFAULT_SIZE), None)
        return analysis.light_curve(None, b, ra=ra, dec=dec, folder=folder)
    except (Exception, SystemExit) as exc:
        _fail(exc)


@app.get("/api/movers")
def movers_api(target: str = None, ra: float = None, dec: float = None, band: str = "D1",
               size: float = Query(0.05, ge=0.02, le=0.2), fetch: bool = True, force: bool = False):
    """Scan the field for moving objects and check each mover against Gaia DR3."""
    try:
        return movers.scan(target, ra, dec, band, size, fetch=fetch, force=force)
    except (Exception, SystemExit) as exc:
        _fail(exc)


@app.get("/api/gaia")
def gaia_api(target: str = None, ra: float = None, dec: float = None, band: str = "D1",
             size: float = Query(0.05, ge=0.02, le=0.2), fetch: bool = True):
    """Gaia DR3 stars in the field, moved to the SPHEREx epoch, with pixel positions on the frames."""
    try:
        b = band.upper()
        return movers.gaia_overlay(target, ra, dec, b, size, fetch=fetch)
    except (Exception, SystemExit) as exc:
        _fail(exc)


# The website itself (must come last so /api/... routes win)
Path("static/frames").mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
