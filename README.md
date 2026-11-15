# SPHEREx Sky Time-Lapse Explorer

**NASA Space Apps Challenge 2026** · a public web tool that turns NASA SPHEREx infrared images into a time-lapse of any patch of sky, so anyone can spot objects that move or change brightness, and then check what those objects are made of.

> In 1930, Pluto was discovered by blinking two photographs of the sky. SPHEREx now photographs the **entire sky every six months in 102 infrared colours**. This tool lets anyone do the same kind of discovery with that data, in a browser.

---

## What it does

Type a target name or sky coordinates (RA/Dec). The tool fetches the SPHEREx images of that spot **live from NASA's IRSA archive**, lines them up, and shows:

| Feature | What you get |
|---|---|
| **Movie** | One clean frame per six-month sky pass, played as an animation |
| **Blink** | Flip between the first and last frame: the classic discovery method |
| **Difference** | Last frame minus first: a moving object appears as a red/blue pair |
| **Deep** | All epochs added together, to reveal faint objects |
| **Motion measurement** | The object's proper motion in arcseconds per year, compared with the reference value from the Gaia mission |
| **Spectrum on the fly** | Click any star: its 0.75–5 µm spectrum is built from every SPHEREx image of it, with water, methane, CO and CO₂ bands marked |
| **Light curve** | Brightness over time for variable stars |
| **Detector choice** | Any of SPHEREx's 6 detector arrays (D1–D6), each a stack of all its images |
| **Field of view** | 3′, 6′ or 12′ (larger fields show more surrounding sky but download more data) |
| **Sky map** | Gaia DR3 stars drawn directly on the SPHEREx image; Aladin Lite with a WISE infrared layer when CDS is online, plus a "Compare in WiseView" link |
| **Data quality** | Saturated pixels detected from SPHEREx's own data and handled automatically |
| **Shareable / embeddable** | Every view is a URL (`#target=barnard&band=D3&mode=blink…`) |
| **RGB colour view** | Three detectors as colours (D6 red, D3 green, D1 blue): cold brown dwarfs stand out in red |
| **Click any star** | Its motion is measured and checked live against Gaia DR3, and its spectrum is built |
| **Scan for moving objects** | One click finds every source, measures each one's motion, checks movers against Gaia, and flags **discovery candidates** |
| **AI assistants (MCP)** | An MCP server exposes the same tools, so assistants like Claude can answer questions with live SPHEREx data |

## Results: does it measure real science?

We tested the tool on the challenge's targets and compared what it measures with independent reference values.

**Proper motion** (one year of SPHEREx data, 3–4 sky passes):

| Target | Detector | SPHEREx (this tool) | Reference | Agreement |
|---|---|---|---|---|
| Barnard's Star | D3 | **10.17 ± 0.44″/yr** | 10.39″/yr (Gaia DR3) | ✅ consistent |
| Epsilon Indi B | D3 | **4.53 ± 0.28″/yr** | 4.71″/yr* | ✅ consistent (0.7σ) |
| Epsilon Indi B | D6 | **4.51 ± 0.28″/yr** | 4.71″/yr* | ✅ consistent (1.0σ) |
| WISE 0855 | D6 (stacked) | **8.96 ± 1.21″/yr** | 8.13″/yr (published, Spitzer) | ✅ consistent |

\*Published value for the Epsilon Indi system. Epsilon Indi B itself is too faint in visible light to be in Gaia.

- **Barnard's Star saturates** SPHEREx's detectors (its brightest pixels are missing). The tool detects this from the data and fits only the unsaturated pixels around the star. A simple centroid gave 8.3″/yr; the saturation-aware fit gives 10.2″/yr.
- For Epsilon Indi B, **two independent detectors agree to 0.02″/yr.**
- **WISE 0855**, the coldest known brown dwarf, is invisible in single SPHEREx images. The tool automatically measures it on the stacked six-month epochs instead, and its motion matches the published value.
- **Gaia misses brown dwarfs:** the scan finds Epsilon Indi B moving, though it is not in Gaia DR3 at all: cold brown dwarfs are too faint in visible light for Gaia but bright in SPHEREx's infrared. This is why SPHEREx can find nearby objects that Gaia cannot.

**Spectrum of Epsilon Indi B** (198 image points from all 6 detectors): deep water absorption at 1.15, 1.4 and 1.9 µm, methane cutting off the 1.6 µm peak, a broad methane trough from 2.2 to 3.6 µm, and a strong emission peak at ~4 µm. This is the signature of a **cold, methane-rich brown dwarf atmosphere**, measured directly from SPHEREx data.

**Known limits (honest list):**
- For saturated stars, a small east-west offset remains (Barnard's Star); fitting SPHEREx's own PSF model is the next step.
- Very faint objects are measured on only a few stacked epochs, so their uncertainties are larger (WISE 0855: ±1.2″/yr); more SPHEREx data will tighten this.
- Candidate movers need checking by eye: blends, artefacts and saturated neighbours can mimic motion.
- Spectrum flux levels are approximate (simple aperture); the *shape* is reliable.

## How it works

```
Browser (HTML/CSS/JS)                Python server (FastAPI)                     NASA / ESA
─────────────────────                ───────────────────────                     ──────────
target / RA,Dec, detector  ──▶  /api/frames   → pipeline.py                ──▶  IRSA: SPHEREx images
movie, blink, diff, deep   ◀──    • search IRSA (TAP), download cutouts         (live, cached 24 h)
                                  • reproject to one north-up sky grid
                                  • group into 6-month epochs, stack (median)
motion card                ──▶  /api/motion   → analysis.py: PSF fit per image,
                                  saturation-aware, weighted fit vs time
spectrum chart (click)     ──▶  /api/spectrum → each image gives one wavelength
                                  (SPHEREx WCS-WAVE map) → 102-channel spectrum
light curve                ──▶  /api/lightcurve
sky map  ─────────────────────────────────────────────────────────────────────▶  CDS Aladin, Gaia DR3, WISE
```

- **Always up to date:** nothing from the archive is stored permanently. Every position is fetched live from IRSA and cached for 24 hours, so new SPHEREx releases appear automatically (the tool already picked up **QR3 and DR1** without code changes).
- **Robust:** failed downloads and searches are retried, with a second IRSA search route; Gaia lookups try four data centres (ESA, ARI Heidelberg, GAVO, NOIRLab); if a service is unavailable, the tool falls back to cached data.

## Run it yourself

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000. The first load of a new position takes 1–3 minutes (live NASA query); after that it is cached.

Optional command-line tools:
```bash
python precache.py                         # prepare all demo targets
python fetch_spherex.py --name "Barnard's Star" --band D3
python analysis.py motion --target barnard --band D3
python analysis.py spectrum --target epsindib
python check_data.py                       # verify downloaded files
```

## API

| Endpoint | Returns |
|---|---|
| `GET /api/targets` | The five built-in test targets |
| `GET /api/frames?target=barnard&band=D3&size=0.05` | Frame URLs, blink/diff/deep images, dates, image counts |
| `GET /api/frames?ra=269.45&dec=4.77&band=D2` | Same, for any position |
| `GET /api/motion?target=epsindib&band=D6` | Measured vs reference proper motion |
| `GET /api/spectrum?target=epsindib[&ra=..&dec=..]` | Spectrum of the target or of any source in its field |
| `GET /api/lightcurve?target=rdra` (or `?ra=..&dec=..`) | Brightness vs time |
| `GET /api/motion?field=Barnards_Star&ra=..&dec=..` | Motion of any star in a field, checked against Gaia |
| `GET /api/movers?target=barnard&band=D3` | All moving objects in the field, with Gaia status |

Interactive docs: http://127.0.0.1:8000/docs

## Project files

| File | Purpose |
|---|---|
| `main.py` | Web server and API |
| `pipeline.py` | Live data access, alignment, stacking, frames |
| `analysis.py` | Motion (saturation-aware PSF fit), spectrum, light curves |
| `fetch_spherex.py` | Find and download SPHEREx cutouts from IRSA |
| `check_data.py` | Verify downloaded files |
| `precache.py` | Prepare the demo targets in one run |
| `movers.py` | Scan a field for moving objects and check them against Gaia |
| `mcp_server.py` | MCP server for AI assistants (Claude Desktop etc.) |
| `static/` | The website (HTML, CSS, JavaScript) |

## Data and tools

- **SPHEREx** spectral images: NASA/JPL-Caltech, served by **NASA/IPAC IRSA** (SIA/TAP search and cutout service).
- **Gaia DR3** (ESA) reference proper motions and sky overlay; **WISE/allWISE** infrared layer.
- **Aladin Lite** (CDS, Strasbourg) for the interactive sky map.
- **Astropy**, **astroquery**, **reproject**, NumPy, SciPy, Matplotlib, FastAPI, Chart.js.
- Inspired by **WiseView** (Backyard Worlds), which volunteers have used to discover brown dwarfs in WISE data.

## Use it from an AI assistant (MCP)

```bash
pip install mcp
python mcp_server.py --selftest      # check all tools on cached data
```
Claude Desktop: add to `claude_desktop_config.json`:
```json
{ "mcpServers": { "spherex": { "command": "<path>/.venv/Scripts/python.exe", "args": ["<path>/mcp_server.py"] } } }
```
Tools: `list_targets`, `spherex_timelapse` (returns the frames as images), `spherex_motion`, `spherex_scan_movers`, `spherex_spectrum`, `spherex_lightcurve`. All accept a target name or any RA/Dec.

## Future work

- Fit SPHEREx's own PSF model for even better positions of saturated stars
- Model parallax in the motion fit (distances for nearby objects)
- Moving solar-system objects via IRSA's Moving Object Search Tool
- Alerts for new discovery candidates as each SPHEREx release arrives

## Team

Beyond Red
