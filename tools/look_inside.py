"""
Look inside a trained adapter.

Reads the saved adapter file directly (no model is loaded, so it takes a few
seconds) and shows what training put there, and how much it changes each part
of the model.

    python tools/look_inside.py                              your Lab 3 adapter
    python tools/look_inside.py out/smollm2-lora out/smollm2-qlora   compare two
"""

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import common.quiet   # hide library warnings and progress bars

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from common.triage import OUT_DIR, REPO_ROOT

GROUP = 4      # layers per bar in the chart
BAR = 30       # longest bar, in characters


def model_file(name):
    """Where the original model's numbers are on this computer, or None."""
    if os.path.isdir(name):
        path = os.path.join(name, "model.safetensors")
        return path if os.path.exists(path) else None
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(name, "model.safetensors")
        return path if isinstance(path, str) else None
    except Exception:
        return None


def read_adapter(folder):
    """Everything the chart and tables need from one saved adapter."""
    with open(os.path.join(folder, "adapter_config.json")) as f:
        cfg = json.load(f)
    tables = load_file(os.path.join(folder, "adapter_model.safetensors"))
    scale = cfg["lora_alpha"] / cfg["r"]       # how strongly the adapter counts
    base = model_file(cfg["base_model_name_or_path"])
    own = safe_open(base, framework="pt") if base else None

    parts = []                                 # one entry per part with an adapter
    for key in tables:
        if ".lora_A." not in key:
            continue
        name = key.split("base_model.model.")[-1].replace(".lora_A.weight", "")
        a = tables[key].float()
        b = tables[key.replace(".lora_A.", ".lora_B.")].float()
        change = (scale * b @ a).norm().item()   # size of what the adapter adds
        size = None                               # size of the part's own numbers
        if own is not None:
            try:
                size = own.get_tensor(name + ".weight").float().norm().item()
            except Exception:
                pass
        layer = int(name.split("layers.")[1].split(".")[0])
        parts.append({"name": name[name.find("layers."):], "layer": layer,
                      "change": change, "size": size, "b": b})

    parts.sort(key=lambda p: p["layer"])       # layer 1 first
    numbers = sum(t.numel() for t in tables.values())
    file_mb = os.path.getsize(os.path.join(folder, "adapter_model.safetensors")) / 1024 ** 2
    return {"folder": os.path.relpath(folder, REPO_ROOT), "cfg": cfg, "parts": parts,
            "numbers": numbers, "file_mb": file_mb, "have_own": own is not None}


def share(p):
    """How big the change is, as a share of the part's own numbers."""
    return p["change"] / p["size"] if p["size"] else None


def by_group(info):
    """Average change for each group of GROUP layers."""
    layers = max(p["layer"] for p in info["parts"]) + 1
    groups = []
    for g in range(0, layers, GROUP):
        ps = [p for p in info["parts"] if g <= p["layer"] < g + GROUP]
        if info["have_own"]:
            value = sum(share(p) for p in ps) / len(ps)
        else:
            value = sum(p["change"] for p in ps) / len(ps)
        groups.append((f"layers {g + 1}-{min(g + GROUP, layers)}", value))
    return layers, groups


def as_text(value, have_own):
    return f"{value:.2%}" if have_own else f"{value:.3f}"


def one(folder):
    info = read_adapter(folder)
    cfg, parts = info["cfg"], info["parts"]
    layers, groups = by_group(info)

    print(f"\nInside your adapter: {info['folder']}\n")
    print(f"  Fits only this model  : {cfg['base_model_name_or_path']}")
    print(f"  Adapter size (r)      : {cfg['r']}    how strongly it counts: {cfg['lora_alpha']}")
    print(f"  Parts with an adapter : {len(parts)}, {len(parts) // layers} in each of "
          f"the model's {layers} layers")
    print(f"  Numbers in it         : {info['numbers']:,}   file: {info['file_mb']:.1f} MB")

    # The same part out/lab3_adapter.txt showed before training.
    first = next((p for p in parts if p["name"] == "layers.0.self_attn.q_proj"), parts[0])
    print(f"\n  Table B started as all zeros (see out/lab3_adapter.txt). Now, the first")
    print(f"  5 numbers of B in {first['name']}:")
    print("    " + "  ".join(f"{v:.4f}" for v in first["b"][:5, 0].tolist()))

    have = info["have_own"]
    print("\n  How much the adapter changes each group of layers"
          + (", as a share of\n  their own numbers" if have else "") +
          f" (layer 1 reads the ticket first, layer {layers} writes the answer):")
    top = max(v for _, v in groups)
    for label, v in groups:
        print(f"    {label:13s} {'#' * max(1, round(BAR * v / top)):{BAR}s}  {as_text(v, have)}")

    if have:
        most = max(parts, key=share)
        least = min(parts, key=share)
        print(f"\n  Most changed part : {most['name']}  ({share(most):.2%})")
        print(f"  Least changed part: {least['name']}  ({share(least):.2%})")
        if share(most) < 0.1:
            print(f"\n  Even the biggest change is {share(most):.1%} of that part's own numbers:"
                  "\n  training nudged the model, it did not rewrite it.")
    else:
        print("\n  (The original model is not on this computer, so changes are shown"
              "\n  as raw sizes, not as a share of the model's own numbers.)")


def compare(folders):
    infos = [read_adapter(f) for f in folders]
    have = all(i["have_own"] for i in infos)
    width = max(14, *(len(i["folder"]) for i in infos))
    print("\nTwo adapters side by side\n")
    print(f"  {'':22s}" + "".join(f"{i['folder']:>{width + 2}s}" for i in infos))
    rows = [("adapter size (r)", lambda i: str(i["cfg"]["r"])),
            ("numbers in it", lambda i: f"{i['numbers']:,}"),
            ("file MB", lambda i: f"{i['file_mb']:.1f}")]
    for label, get in rows:
        print(f"  {label:22s}" + "".join(f"{get(i):>{width + 2}s}" for i in infos))

    print("\n  How much each changes the model"
          + (", as a share of its own numbers:" if have else ":"))
    groups = [by_group(i)[1] for i in infos]
    for k, (label, _) in enumerate(groups[0]):
        print(f"  {label:22s}" + "".join(
            f"{as_text(g[k][1], have):>{width + 2}s}" for g in groups))


folders = sys.argv[1:] or [os.path.join(OUT_DIR, "smollm2-lora")]
for f in folders:
    if not os.path.exists(os.path.join(f, "adapter_model.safetensors")):
        raise SystemExit(f"\n  No adapter in {f}. Run the lab that saves it first.")
if len(folders) == 1:
    one(folders[0])
else:
    compare(folders)
