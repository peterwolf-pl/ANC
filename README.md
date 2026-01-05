# ANC plotter toolkit

Python utilities for generating and working with **ACODE** instructions for a differential-drive pen plotter. The tools convert CAD or raster inputs into ACODE, preview the resulting motion, and optionally stream the commands to an ESP32-based controller or a small Flask UI.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional: dependencies for the Flask UI
pip install -r requirements-ui.txt
```

The scripts are standalone; install only the packages you need for the commands you plan to run (for example, `ezdxf` for DXF conversion or `matplotlib` for visualization).

## Tools and quick starts

### DXF → ACODE (`acode.py`)
- Converts DXF geometry to ACODE with one-shot arcs and LWPOLYLINE bulge support.
- Supports filtering by `--layer`, controlling feedrates (`--feed-lin`, `--feed-turn`, `--feed-arc`), and basic simplification (`--flat-step`, `--epsilon`).
- Example: `python acode.py drawing.dxf --layer art -o drawing.acode`

### PNG → ACODE (`acodepng.py`)
- Builds horizontal scanlines from a bitmap using configurable spacing and sampling.
- Ordering can favor nearest paths or deterministic scanlines (`--path-order` + `--scan-direction`).
- Example: `python acodepng.py input.png --img-width-mm 180 --path-order scanline -o input.acode`

### Art-focused PNG → ACODE (`artacodepng.py`)
- Higher-fidelity scanline converter with jitter, serpentine traversal, and multiple row-advance strategies.
- Useful flags: `--line-spacing-mm`, `--x-mode` (`pixel` vs `step`), `--line-advance` (`soft`, `turn90`, `real90`).
- Example: `python artacodepng.py posterized.png --img-width-mm 200 --line-spacing-mm 0.7 -o posterized.acode`

### Visualize ACODE (`acodeviz.py`)
- Renders the pen-down path to PNG/SVG/PDF and reports stats like final pose and bounding box.
- Example: `python acodeviz.py drawing.acode --equal --invert-y -o preview.png`

### Stream ACODE to the plotter (`print.py`)
- Sends ACODE over TCP to the ESP32 firmware, printing status per line.
- Configure the target with `--host`, `--port`, and motion parameters (`--feed-to-sps`, `--min-sps`, `--max-sps`, etc.).
- Example: `python print.py drawing.acode --host 192.168.4.1 --port 3333`

### Web UI (`ui_sender.py`)
python3 ui_sender.py
- Flask app combining a sender, PNG→ACODE generator, and simulator overlay in one interface.
- Writes temporary files to `_ui_work/` and calls `artacodepng.py`/`acodeviz.py` internally.
- Start with `python ui_sender.py` and open http://127.0.0.1:5000/.

## Typical workflow
1. Prepare input artwork (DXF or PNG) and convert it to ACODE with `acode.py`, `acodepng.py`, or `artacodepng.py`.
2. Preview the motion and bounding box with `acodeviz.py` to confirm orientation and scale.
3. Stream the verified ACODE to the plotter using `print.py` or the web UI.

OR 
USE WEB UI 
