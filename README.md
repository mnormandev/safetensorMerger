# safetensorMerger

Combines sharded model files (for example `model-00001-of-00002.safetensors` and `model-00002-of-00002.safetensors`) into **one** `.safetensors` file, e.g. for ComfyUI's **Load CLIP** node.

- Works on **macOS, Windows and Linux**
- **No dependencies**: just Python 3, no torch or pip installs
- Works with any data type (fp8, bf16, fp16, fp32)
- Streams data straight from disk to disk, so it **doesn't load the model into RAM**
- No hardcoded paths: you drag the files into the terminal window

---

## Requirements

Python 3.8 or newer. Check with:

```bash
python3 --version
```

macOS: if it's missing, run `xcode-select --install` or install it from [python.org](https://www.python.org/downloads/).

---

## Running it

### macOS

**Double-click `run.command`** in Finder. A Terminal window opens with the merger running.

The first time, you may need to make it executable:

```bash
cd ~/www/tensorMerger
chmod +x run.command
```

If macOS blocks it, right-click `run.command` → **Open** → **Open**.

Or run it from Terminal:

```bash
cd ~/www/tensorMerger
python3 merge_safetensors.py
```

### Windows

Double-click **`run.bat`**, or run:

```bat
python merge_safetensors.py
```

### Linux

```bash
python3 merge_safetensors.py
```

---

## Step by step

### 1. Add the shard files

Drag a shard file from Finder / Explorer into the terminal window and press **Enter**. Repeat for each shard.

You can also:

- drop **several files at once**
- drop a **whole folder** to add every `.safetensors` file inside it

After each drop, the current list is shown:

```
  1. model-00001-of-00002.safetensors  [3.9 GB]
  2. model-00002-of-00002.safetensors  [1.2 GB]
```

Commands while adding files:

| Key | Action |
|-----|--------|
| `u` | Remove the last file added |
| `c` | Clear the whole list |
| `q` | Quit |

Files that aren't `.safetensors` (such as `model.safetensors.index.json`), files that can't be read, and duplicates are skipped automatically.

### 2. Confirm the list

When all files are in the list, press **Enter** on an empty line, or type **`Y`** and press Enter.

### 3. Name the output file

By default the output is saved in the **tensorMerger folder**, next to the script. Just type a name:

```
Output file name (or drag a different folder here) > qwen3vl_4b_krea2_abliterated_fp8
```

- The `.safetensors` extension is added automatically if you leave it out.
- To save somewhere else, **drag a folder** in first, then type the name. For example, drag in `ComfyUI/models/text_encoders` to save it straight where ComfyUI looks.
- You can also type or paste a full path.
- If the file already exists, you're asked before it's overwritten.

### 4. Wait for the merge

A progress percentage is shown while it writes. When it prints `Done`, the file is ready.

The file is written as `<name>.safetensors.part` first and renamed at the end, so a cancelled or failed merge never leaves a broken model file behind.

---

## Using the result in ComfyUI

1. Move the merged file into `ComfyUI/models/text_encoders/` (skip this if you saved it there directly).
2. Refresh ComfyUI: press **R** in the canvas, or restart it.
3. Select the file in the **Load CLIP** node.

---

## Notes

- **Disk space:** you need free space equal to the combined size of the shards.
- **Duplicate tensor names:** if two shards contain a tensor with the same name, you get a warning and the later file's version is used. This shouldn't happen with normal shards.
- **Shard order** doesn't matter. Each tensor keeps its own name.
- **Metadata** from the shards is kept, and `format: pt` is added if missing.
- Merged `.safetensors` files are listed in `.gitignore`, so outputs saved in this folder won't be committed.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `run.command` won't open | Run `chmod +x run.command`, then right-click → **Open**. |
| `python3: command not found` | Install Python 3 (see Requirements). |
| `Not found:` after dropping a file | Make sure the file still exists. Try dropping it again, or type the path in quotes. |
| `Skipped (can't read ...)` | The file is incomplete or not a real safetensors file. Download it again. |
| `Folder doesn't exist` | The folder in your output path is wrong. Drag the target folder into the window instead. |
| The file doesn't appear in Load CLIP | Check it's in `models/text_encoders/`, then refresh ComfyUI. |
| Cancel at any point | Press **Ctrl + C**. |

---

## Project files

| File | Purpose |
|------|---------|
| `merge_safetensors.py` | The merger |
| `run.command` | Double-click launcher for macOS |
| `run.bat` | Double-click launcher for Windows |
| `.gitignore` | Keeps model files and Python cache out of git |
