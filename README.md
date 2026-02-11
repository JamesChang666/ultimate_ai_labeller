# Ultimate AI Labeller

Desktop image-annotation tool for computer vision datasets, built with Tkinter.

## Features

- Fast bounding-box labeling with mouse drag, resize handles, and move operations
- Keyboard-first workflow (`F` next, `D` previous, `A` auto red-detection, `Space` fuse boxes)
- Undo/redo support (`Ctrl+Z`, `Ctrl+Y`)
- YOLO-assisted detection (Ultralytics) with confidence control
- Box fusion utilities for overlapping/nearby boxes
- Scrollable right settings panel for small window sizes
- Remove bad frames from current split (`train`/`val`/`test`)
- Restore removed frames from a selection dialog
- Image jump selector (dropdown) to move directly to any image in the split
- Session resume memory (reopens last project/split/image)
- Language and theme switch support in-app

## Dataset Layout

The app expects a YOLO-style folder structure:

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

Supported image extensions: `.png`, `.jpg`, `.jpeg`  
Label format: YOLO `.txt` (`class cx cy w h` normalized).

Removed frames are moved to:

```text
your_project/
  removed/
    train|val|test/
      images/
      labels/
```

## Install

```bash
pip install -e .
```

or:

```bash
pip install .
```

## Run

After install:

```bash
ai-labeller
```

Or directly:

```bash
python src/ai_labeller/main.py
```

## Shortcuts

- `F`: save and next image
- `D`: previous image
- `A`: auto red detection
- `Space`: fuse boxes
- `Ctrl+Z`: undo
- `Ctrl+Y`: redo
- `Delete`: delete selected box

## Notes

- First YOLO use requires model weights (default: `yolov8n.pt`) available in your working directory or configured path.
- Session state is saved to `~/.ai_labeller_session.json`.
