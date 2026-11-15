"""
mcp_server.py - lets AI assistants (Claude Desktop and other MCP clients) use the
SPHEREx Sky Time-Lapse Explorer.

The server wraps the same functions as the website, so every answer comes from live
NASA SPHEREx data (with the same 24 h cache), checked against Gaia where relevant.

Tools
  list_targets             the built-in test targets
  spherex_timelapse        aligned frames for a position: dates, image counts, and the
                           first / last / difference images so the assistant can SEE them
  spherex_motion           proper motion of a target or of any star, vs Gaia DR3
  spherex_scan_movers      every moving object in a field, with discovery candidates
  spherex_spectrum         0.75-5 micron spectrum, with molecular bands
  spherex_lightcurve       brightness over time (variable stars)

Run
  pip install mcp
  python mcp_server.py --selftest      # quick check without any AI client
  python mcp_server.py                 # stdio mode (Claude Desktop starts it for you)
  python mcp_server.py --http          # HTTP mode on port 8001 (for remote MCP clients)

Claude Desktop: add this to %APPDATA%\\Claude\\claude_desktop_config.json (Windows), then restart Claude:
  {
    "mcpServers": {
      "spherex": {
        "command": "D:\\\\NASA SPACE APPS CHALLENGE\\\\.venv\\\\Scripts\\\\python.exe",
        "args": ["D:\\\\NASA SPACE APPS CHALLENGE\\\\mcp_server.py"]
      }
    }
  }
"""

import contextlib
import functools
import os
import sys
from pathlib import Path
from typing import Optional

# Claude Desktop starts us from another folder; our code uses relative paths (data/, static/).
os.chdir(Path(__file__).resolve().parent)

# Works with both MCP SDK versions: 2.x renamed FastMCP to MCPServer.
try:
    from mcp.server.mcpserver import Image, MCPServer as _Server  # noqa: E402  (mcp 2.x)
    MCP_V2 = True
except ImportError:
    from mcp.server.fastmcp import FastMCP as _Server, Image  # noqa: E402  (mcp 1.x)
    MCP_V2 = False

import analysis  # noqa: E402
import movers  # noqa: E402
import pipeline  # noqa: E402
from fetch_spherex import PRESETS  # noqa: E402

BEST_BAND = {"barnard": "D3", "epsindib": "D6", "wise0855": "D6", "rwaur": "D1", "rdra": "D1"}
ALL_BANDS = ("D1", "D2", "D3", "D4", "D5", "D6")

mcp = _Server(
    "spherex-explorer",
    instructions=(
        "Tools for NASA SPHEREx all-sky infrared images (0.75-5 micron, 6 detectors D1-D6, the whole sky "
        "every six months since 2025). Use them to see how a patch of sky changes over time, measure how "
        "fast stars move (checked against ESA Gaia), find moving objects, and build infrared spectra. "
        "Positions are RA/Dec in degrees. Built-in targets: barnard, epsindib, wise0855, rwaur, rdra. "
        "Data comes live from NASA/IPAC IRSA; a first request for a new position can take 1-3 minutes."
    ),
)


def quiet(fn):
    """MCP over stdio uses stdout for protocol messages, so send our progress prints to stderr."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with contextlib.redirect_stdout(sys.stderr):
            try:
                return fn(*args, **kwargs)
            except SystemExit as exc:  # our CLI helpers use SystemExit for user-facing errors
                raise ValueError(str(exc)) from None
    return wrapper


def _band(target, band):
    if band:
        return band.upper()
    key = pipeline.resolve(target)[0] if target else None
    return BEST_BAND.get(key, "D1")


def _png_path(url):
    """Turn a website URL like /frames/X/e1.png into the file on disk."""
    return Path("static") / url.lstrip("/")


@mcp.tool()
@quiet
def list_targets() -> list[dict]:
    """List the built-in test targets (name, key, best detector, kind of variability)."""
    kinds = {"rwaur": "variable star", "rdra": "variable star"}
    return [{"key": k, "name": p["label"].replace("_", " "), "best_band": BEST_BAND.get(k, "D1"),
             "kind": kinds.get(k, "moving object")} for k, p in PRESETS.items()]


@mcp.tool()
@quiet
def spherex_timelapse(target: str = "", ra: Optional[float] = None, dec: Optional[float] = None,
                      band: str = "", size_deg: float = 0.05) -> list:
    """
    Build a SPHEREx time-lapse of a sky position and return a summary plus three images:
    the first epoch, the last epoch, and the difference (last minus first; a moving object
    appears as a red/blue pair). Give either a built-in target key, or ra and dec in degrees.
    band: D1..D6 or RGB (default: best band for the target, else D1). size_deg: 0.05-0.2.
    """
    b = _band(target, band)
    m = pipeline.build_frames(target or None, ra, dec, b, size_deg)
    summary = {
        "target": m["target"]["label"], "ra": m["target"]["ra"], "dec": m["target"]["dec"],
        "band": m["band"], "n_images": m["n_images_total"], "n_epochs": m["n_epochs"],
        "latest_image_date": m["latest_image_date"],
        "epochs": [{"date": f["date"], "n_images": f["n_images"]} for f in m["frames"]],
        "images_returned": ["first epoch", "last epoch", "difference (last - first)"],
        "orientation": "north up, east left",
        "pixel_scale": ("images are resampled onto a 3 arcsec/pixel grid (SPHEREx's own detector "
                        "pixels are about 6.2 arcsec) and then enlarged 4x for display"),
    }
    out = [summary]
    for url in (m["frames"][0]["url"], m["frames"][-1]["url"], m["diff_url"]):
        p = _png_path(url)
        if p.exists():
            out.append(Image(path=str(p)))
    return out


@mcp.tool()
@quiet
def spherex_motion(target: str = "", ra: Optional[float] = None, dec: Optional[float] = None,
                   band: str = "") -> dict:
    """
    Measure proper motion (arcsec/year) from SPHEREx images and compare it with a reference.
    - target only (e.g. "barnard"): the built-in target, compared with its known value.
    - ra/dec (degrees): ANY star; the reference comes from a live Gaia DR3 lookup. A clearly
      moving object with no Gaia motion is reported as a discovery candidate.
    """
    if ra is not None and dec is not None:
        b = _band(target, band)
        if target:
            key, label, _, _ = pipeline.resolve(target)
            folder = pipeline.data_folder(label, pipeline.DEFAULT_SIZE)
            pipeline.ensure_data(target, bands=(b,))
        else:
            folder, _ = pipeline.ensure_data(None, ra, dec, bands=(b,))
        r = analysis.measure_motion(None, b, min_snr=5.0, make_plot=False, ra=ra, dec=dec, folder=folder)
    else:
        if not target:
            raise ValueError("Give a target key, or ra and dec.")
        b = _band(target, band)
        pipeline.ensure_data(target, bands=(b,))
        r = analysis.measure_motion(target, b, make_plot=False)
    keep = ["target", "star", "band", "measured", "reference", "reference_note", "verdict",
            "difference_sigma", "n_used", "n_epochs", "baseline_years", "n_core_missing", "method"]
    return {k: r.get(k) for k in keep}


@mcp.tool()
@quiet
def spherex_scan_movers(target: str = "", ra: Optional[float] = None, dec: Optional[float] = None,
                        band: str = "", size_deg: float = 0.05) -> dict:
    """
    Find every source in a SPHEREx field, measure how each moves between the six-month sky
    passes, and check every mover against Gaia DR3. Statuses: "known mover", "known target",
    "DISCOVERY CANDIDATE" (moving but not in Gaia), "unverified (Gaia offline)".
    Candidates still need checking by eye: blends and artefacts can mimic motion.
    """
    b = _band(target, band)
    if b == "RGB":
        b = "D3"
    r = movers.scan(target or None, ra, dec, b, size_deg)
    mv = [{k: s.get(k) for k in ("ra", "dec", "total", "total_err", "pmra", "pmdec", "snr", "status", "gaia")}
          for s in r["sources"] if s["moving"]]
    return {"field": r["target"]["label"], "band": r["band"], "n_sources": r["n_sources"],
            "n_movers": r["n_movers"], "n_candidates": r["n_candidates"],
            "gaia_checked": r["gaia_checked"], "movers": mv, "note": r["note"]}


@mcp.tool()
@quiet
def spherex_spectrum(target: str = "", ra: Optional[float] = None, dec: Optional[float] = None) -> dict:
    """
    Build a 0.75-5 micron spectrum from every SPHEREx image of a source (each image gives one
    wavelength). Returns ~100 channels plus the expected positions of water, methane, CO and CO2
    bands. Give a built-in target (optionally with ra/dec of another source in its field), or ra/dec.
    Fluxes are approximate (simple aperture); the SHAPE is what matters.
    """
    if target:
        pipeline.ensure_data(target, bands=ALL_BANDS)
        s = analysis.get_spectrum(target, ra, dec, make_plot=False)
    else:
        if ra is None or dec is None:
            raise ValueError("Give a target key, or ra and dec.")
        folder, _ = pipeline.ensure_data(None, ra, dec, bands=ALL_BANDS)
        s = analysis.get_spectrum(None, ra, dec, folder=folder, make_plot=False)
    peak = max(range(len(s["flux"])), key=lambda i: s["flux"][i]) if s["flux"] else None
    return {"source": s["target"], "unit": s["unit"], "n_points": s["n_points"], "bands_used": s["bands_used"],
            "brightest_at_um": None if peak is None else s["wavelength_um"][peak],
            "wavelength_um": s["wavelength_um"], "flux": s["flux"], "features": s["features"]}


@mcp.tool()
@quiet
def spherex_lightcurve(target: str = "", ra: Optional[float] = None, dec: Optional[float] = None,
                       band: str = "D1") -> dict:
    """
    Brightness of a source in every SPHEREx image over time (variable stars, e.g. rwaur, rdra).
    Give a built-in target key, or ra and dec in degrees for any source.
    """
    b = band.upper()
    if ra is not None and dec is not None and not target:
        folder, _ = pipeline.ensure_data(None, ra, dec, bands=(b,))
        return analysis.light_curve(None, b, make_plot=False, ra=ra, dec=dec, folder=folder)
    if not target:
        raise ValueError("Give a target key, or ra and dec.")
    pipeline.ensure_data(target, bands=(b,))
    if ra is not None and dec is not None:
        return analysis.light_curve(target, b, make_plot=False, ra=ra, dec=dec)
    return analysis.light_curve(target, b, make_plot=False)


def selftest():
    """Call each tool on cached data and print short results (no AI client needed)."""
    def show(name, fn):
        print(f"\n== {name}")
        try:
            r = fn()
            if isinstance(r, list):
                print(f"   summary: {r[0]}\n   images: {len(r) - 1}")
            elif isinstance(r, dict):
                for k, v in list(r.items())[:8]:
                    print(f"   {k}: {str(v)[:110]}")
            else:
                print("  ", str(r)[:300])
        except Exception as exc:
            print(f"   FAILED: {type(exc).__name__}: {exc}")
    show("list_targets", list_targets)
    show("spherex_timelapse(barnard)", lambda: spherex_timelapse("barnard"))
    show("spherex_motion(epsindib)", lambda: spherex_motion("epsindib"))
    show("spherex_scan_movers(epsindib)", lambda: spherex_scan_movers("epsindib"))
    show("spherex_spectrum(epsindib)", lambda: spherex_spectrum("epsindib"))
    show("spherex_lightcurve(rdra)", lambda: spherex_lightcurve("rdra"))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    elif "--http" in sys.argv:
        if MCP_V2:
            mcp.run(transport="streamable-http", host="127.0.0.1", port=8001)
        else:
            mcp.settings.host = "127.0.0.1"
            mcp.settings.port = 8001
            mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")
