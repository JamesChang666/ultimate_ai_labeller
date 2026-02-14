# Ultimate AI Labeller

Desktop image annotation tool for object detection datasets (Tkinter + Ultralytics).

## Features

- Bounding-box annotation with drag, move, and resize handles
- Undo/redo history (`Ctrl+Z`, `Ctrl+Y`)
- Image navigation (`F` next/save, `D` previous)
- Auto red-region proposal (`A`)
- YOLO detection from UI (`Run Detection`)
- Model source selection:
  - Official bundled `yolo26n.pt`
  - Custom YOLO weights (`.pt`)
  - Custom RF-DETR weights (via Ultralytics interface)
- Auto-detect and propagate options
- Scrollable right settings panel
- Remove/restore bad frames from split
- Image dropdown jump
- Session resume (last project/split/image/model settings)
- English/Chinese UI switch and light/dark theme

## Dataset Structure

```text
your_project/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

- Image extensions: `.png`, `.jpg`, `.jpeg`
- Label format: YOLO txt (`class cx cy w h`, normalized)

Removed frames are moved to:

```text
your_project/
  removed/
    train|val|test/
      images/
      labels/
```

## Install

From PyPI:

```bash
pip install ultimate_ai_labeller
```

From local wheel:

```bash
pip install dist/ultimate_ai_labeller-0.1.4-py3-none-any.whl
```

From source:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Run

```bash
ai-labeller
```

Or:

```bash
python src/ai_labeller/main.py
```

## Shortcuts

- `F`: save and next image
- `D`: previous image
- `A`: auto red detection
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo
- `Delete`: delete selected box

## Notes

- Default detection model mode is `Official YOLO26n.pt (Bundled)`.
- The package includes `yolo26n.pt`, so first-run model download is not required.
- To use your own Tk app icon, put `app_icon.png` in `src/ai_labeller/assets/`.
- Session file: `~/.ai_labeller_session.json`.
