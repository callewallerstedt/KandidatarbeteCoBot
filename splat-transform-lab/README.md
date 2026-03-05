# splat-transform-lab

Quick local setup for converting XGRIDS `.lcc` with PlayCanvas `splat-transform`.

## Why this exists
SuperSplat often fails with raw LCC manifest links (404 on missing chunk files / unsupported URL layout).
Use local conversion first, then open converted output.

## Install (one-time)

### Windows (PowerShell)
```powershell
npm install -g @playcanvas/splat-transform
splat-transform --version
```

### Linux/macOS
```bash
npm install -g @playcanvas/splat-transform
splat-transform --version
```

## GUI (Windows-friendly)

Double-click:
- `00_RUN_LCC_GUI.bat`

Or run manually:
```powershell
python lcc_to_ply_gui.py
```

In GUI:
1. Select `.lcc` input
2. Select `.ply` output
3. Click **Convert LCC → PLY**
4. Watch live log/progress

> GUI uses `npx @playcanvas/splat-transform ...` internally.

## Convert LCC -> outputs

### 1) LCC -> standalone HTML viewer (quick sanity check)
```bash
splat-transform -w input.lcc output.html
```

### 2) LCC -> PLY (for tools that expect geometry-like point data)
```bash
splat-transform -w input.lcc output.ply
```

### 3) LCC -> SOG (PlayCanvas compressed format)
```bash
splat-transform -w input.lcc output.sog
```

### 4) LCC -> unbundled SOG folder
```bash
splat-transform -w input.lcc output/meta.json
```

## Useful cleanup filters

```bash
# remove NaN/Inf
splat-transform -w input.lcc --filter-nan cleaned.ply

# crop by bounding box (min x,y,z, max X,Y,Z)
splat-transform -w input.lcc --filter-box -5,-5,-2,5,5,3 cropped.ply

# keep only most visible splats (e.g., top 30%)
splat-transform -w input.lcc --filter-visibility 30% visible.ply
```

## Notes for your current 404
- Your file looked like an LCC manifest JSON (metadata), not a complete directly-viewable splat payload.
- Convert locally using commands above, then load the converted file.

## Optional GPU selection
```bash
splat-transform --list-gpus
splat-transform -g cpu -w input.lcc output.sog
```
