#!/usr/bin/env python3
"""
tensorMerger - merge sharded .safetensors files into ONE .safetensors file
(e.g. for ComfyUI's "Load CLIP" / "Load Diffusion Model" nodes).

No dependencies: uses only the Python standard library. Tensors are copied
straight from disk to disk, so it works for any dtype (fp8, bf16, ...) and
doesn't need to load the whole model into RAM.

Works on macOS, Linux and Windows.

How to use:
  1. Run it:  python3 merge_safetensors.py   (or double-click run.command on a Mac)
  2. Drag the shard files from Finder / Explorer into the terminal window,
     one at a time (or several at once), pressing Enter after each drop.
     You can also drop a whole folder to add every .safetensors file inside it.
  3. When the list looks right, press Enter on an empty line (or type Y).
  4. Type a name for the output file. It is saved next to this script by
     default; drag a folder in first to save it somewhere else.

Commands while adding files:
  u = remove the last file    c = clear the list    q = quit
"""

import json
import os
import shlex
import struct
import sys

# Importing readline gives line editing and, on macOS, lifts the terminal's
# ~1024 character input limit so long dragged paths aren't cut off.
try:
    import readline  # noqa: F401
except ImportError:
    pass


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IS_WINDOWS = os.name == "nt"
EXT = ".safetensors"
COPY_CHUNK = 64 * 1024 * 1024  # 64 MB


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def split_dropped(text):
    """Turn whatever the terminal pasted on a drop into a list of paths.

    macOS Terminal/iTerm escape spaces with backslashes (My\\ File.safetensors)
    or wrap paths in quotes; Windows wraps paths with spaces in double quotes.
    """
    text = text.strip()
    if not text:
        return []
    try:
        if IS_WINDOWS:
            parts = [p.strip('"').strip("'") for p in shlex.split(text, posix=False)]
        else:
            parts = shlex.split(text)
    except ValueError:
        # Unbalanced quotes etc. - treat the whole line as one path
        parts = [text.strip('"').strip("'")]
    return [os.path.expanduser(p) for p in parts if p]


def human_size(num_bytes):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024


def read_header(path):
    """Return (header_dict, data_start_offset) for a .safetensors file."""
    with open(path, "rb") as f:
        raw_len = f.read(8)
        if len(raw_len) != 8:
            raise ValueError("file is too small to be a safetensors file")
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len > 100 * 1024 * 1024:
            raise ValueError("header is unreasonably large - not a safetensors file?")
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


# ---------------------------------------------------------------------------
# Step 1: collect the shard files
# ---------------------------------------------------------------------------

def print_list(shards):
    print()
    if not shards:
        print("  (no files added yet)")
    for i, path in enumerate(shards, 1):
        print(f"  {i}. {os.path.basename(path)}  [{human_size(os.path.getsize(path))}]")
    print()


def add_path(path, shards):
    path = os.path.abspath(path)

    if os.path.isdir(path):
        found = sorted(
            os.path.join(path, f) for f in os.listdir(path)
            if f.lower().endswith(EXT)
        )
        if not found:
            print(f"  No {EXT} files in folder: {path}")
        for f in found:
            add_path(f, shards)
        return

    if not os.path.isfile(path):
        print(f"  Not found: {path}")
    elif not path.lower().endswith(EXT):
        print(f"  Skipped (not a {EXT} file): {os.path.basename(path)}")
    elif path in shards:
        print(f"  Already added: {os.path.basename(path)}")
    else:
        try:
            read_header(path)
        except Exception as e:
            print(f"  Skipped (can't read {os.path.basename(path)}: {e})")
            return
        shards.append(path)
        print(f"  + {os.path.basename(path)}")


def collect_shards():
    shards = []
    print("=" * 64)
    print(" tensorMerger - merge .safetensors shards")
    print("=" * 64)
    print("Drag shard files (or a folder) into this window and press Enter.")
    print("Press Enter on an empty line or type Y when you're done.")
    print("Commands:  u = undo last   c = clear   q = quit")

    while True:
        raw = input(f"\n[{len(shards)} file(s)] Drop file here > ").strip()
        cmd = raw.lower()

        if cmd in ("", "y", "yes"):
            if not shards:
                print("  Add at least one file first.")
                continue
            if len(shards) == 1:
                print("  Note: only one file added - the output will just be a copy of it.")
            return shards

        if cmd in ("q", "quit", "exit"):
            sys.exit(0)

        if cmd in ("u", "undo"):
            if shards:
                print(f"  - removed {os.path.basename(shards.pop())}")
            print_list(shards)
            continue

        if cmd in ("c", "clear"):
            shards.clear()
            print("  List cleared.")
            continue

        for path in split_dropped(raw):
            add_path(path, shards)
        print_list(shards)


# ---------------------------------------------------------------------------
# Step 2: ask where to save
# ---------------------------------------------------------------------------

def ask_output_path(shards):
    out_dir = SCRIPT_DIR

    while True:
        print(f"\nSave folder: {out_dir}")
        raw = input("Output file name (or drag a different folder here) > ").strip()
        if not raw:
            print("  Please type a file name.")
            continue

        parts = split_dropped(raw)
        # An unquoted name with spaces gets split - put it back together
        target = " ".join(parts) if parts else raw

        # A folder was dropped/typed -> make it the save folder, then ask for a name
        if os.path.isdir(target):
            out_dir = os.path.abspath(target)
            continue

        # Bare name -> save in the current save folder
        if not os.path.dirname(target):
            target = os.path.join(out_dir, target)
        target = os.path.abspath(target)

        if not target.lower().endswith(EXT):
            target += EXT

        if not os.path.isdir(os.path.dirname(target)):
            print(f"  Folder doesn't exist: {os.path.dirname(target)}")
            continue

        if target in shards:
            print("  That's one of the input files - pick a different name.")
            continue

        if os.path.exists(target):
            answer = input(f"  {os.path.basename(target)} already exists. Overwrite? [y/N] > ")
            if answer.strip().lower() not in ("y", "yes"):
                continue

        return target


# ---------------------------------------------------------------------------
# Step 3: merge
# ---------------------------------------------------------------------------

def plan_merge(shards):
    """Read every header and decide which bytes go where in the output.

    Returns (new_header, copy_plan) where copy_plan is a list of
    (source_path, absolute_start, length) in output order.
    """
    tensors = {}   # name -> (path, abs_start, length, dtype, shape)
    metadata = {}

    print()
    for i, path in enumerate(shards, 1):
        header, data_start = read_header(path)
        file_meta = header.pop("__metadata__", None) or {}
        metadata.update(file_meta)

        duplicates = tensors.keys() & header.keys()
        if duplicates:
            print(f"  Warning: {len(duplicates)} tensor name(s) in {os.path.basename(path)} "
                  f"also exist in an earlier file; this file's version will be used.")

        for name, info in header.items():
            begin, end = info["data_offsets"]
            tensors[name] = (path, data_start + begin, end - begin,
                             info["dtype"], info["shape"])

        print(f"[{i}/{len(shards)}] {os.path.basename(path)}: {len(header)} tensors")

    metadata.setdefault("format", "pt")
    new_header = {"__metadata__": {k: str(v) for k, v in metadata.items()}}
    copy_plan = []
    offset = 0
    # Keep each source file's tensors together and in on-disk order for fast copying
    for name, (path, start, length, dtype, shape) in sorted(
            tensors.items(), key=lambda kv: (shards.index(kv[1][0]), kv[1][1])):
        new_header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [offset, offset + length]}
        copy_plan.append((path, start, length))
        offset += length

    return new_header, copy_plan, offset


def merge(shards, output_file):
    new_header, copy_plan, data_size = plan_merge(shards)

    header_bytes = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Pad with spaces so tensor data starts on an 8-byte boundary
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)

    print(f"\nWriting {len(copy_plan)} tensors ({human_size(data_size)}) to:\n  {output_file}")

    tmp_file = output_file + ".part"
    written = 0
    last_pct = -1
    open_sources = {}
    try:
        with open(tmp_file, "wb") as out:
            out.write(struct.pack("<Q", len(header_bytes)))
            out.write(header_bytes)

            for path, start, length in copy_plan:
                src = open_sources.get(path)
                if src is None:
                    for f in open_sources.values():
                        f.close()
                    open_sources = {path: open(path, "rb")}
                    src = open_sources[path]
                src.seek(start)
                remaining = length
                while remaining:
                    chunk = src.read(min(COPY_CHUNK, remaining))
                    if not chunk:
                        raise IOError(f"unexpected end of file in {path}")
                    out.write(chunk)
                    remaining -= len(chunk)
                    written += len(chunk)
                    pct = int(written * 100 / data_size) if data_size else 100
                    if pct != last_pct:
                        print(f"\r  {pct:3d}%  {human_size(written)}", end="", flush=True)
                        last_pct = pct
        print()
        os.replace(tmp_file, output_file)
    except BaseException:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        raise
    finally:
        for f in open_sources.values():
            f.close()

    print(f"\nDone. Output size: {human_size(os.path.getsize(output_file))}")
    print("If it's a text encoder, put it in ComfyUI/models/text_encoders, "
          "refresh ComfyUI (press R) and pick it in Load CLIP.")


def main():
    shards = collect_shards()
    print("\nFiles to merge:")
    print_list(shards)
    output_file = ask_output_path(shards)
    merge(shards, output_file)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        sys.exit(1)
