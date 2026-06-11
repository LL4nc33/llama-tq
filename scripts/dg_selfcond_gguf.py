#!/usr/bin/env python3
"""
GGUF helpers for the self_cond QAT distillation pipeline.

  extract:  pull self_cond_{gate,up,down}.weight from a GGUF (dequantized to f32 .npy)
            — used to get the BF16 teacher's starting weights for training.
  merge:    write trained .npy self_cond weights back INTO a copy of an IQ2 GGUF,
            replacing those 3 tensors (kept at their original quant type, re-quantized
            from the trained f32 via gguf-py).

Usage:
  python dg_selfcond_gguf.py extract --gguf teacher.gguf --out teacher_sc
  python dg_selfcond_gguf.py merge --gguf iq2.gguf --weights trained_sc --out iq2_qat.gguf
"""
import argparse, sys
import numpy as np

try:
    from gguf import GGUFReader, GGUFWriter, dequantize
    import gguf as gguf_mod
except ImportError:
    sys.exit("need gguf-py: pip install gguf  (or use the repo's gguf-py)")

SC_NAMES = ["self_cond_gate.weight", "self_cond_up.weight", "self_cond_down.weight"]
SHORT = {"self_cond_gate.weight": "gate", "self_cond_up.weight": "up", "self_cond_down.weight": "down"}


def extract(args):
    r = GGUFReader(args.gguf)
    found = {}
    for t in r.tensors:
        if t.name in SC_NAMES:
            # dequantize to f32 regardless of stored type
            arr = dequantize(t.data, t.tensor_type) if t.tensor_type.name not in ("F32", "F16", "BF16") else t.data.astype(np.float32)
            arr = np.asarray(arr, dtype=np.float32).reshape(tuple(int(x) for x in t.shape[::-1]))  # gguf shape is reversed
            found[t.name] = arr
            print(f"  {t.name}: type={t.tensor_type.name} shape={arr.shape}")
    if len(found) != 3:
        sys.exit(f"expected 3 self_cond tensors, found {len(found)}: {list(found)}")
    for n, arr in found.items():
        np.save(f"{args.out}.{SHORT[n]}.npy", arr)
    print(f"extracted -> {args.out}.{{gate,up,down}}.npy")


def merge(args):
    """Rebuild the GGUF with the 3 self_cond tensors replaced by the trained f32 weights,
    re-quantized to each tensor's ORIGINAL stored type. All other tensors copied verbatim."""
    r = GGUFReader(args.gguf)
    trained = {
        "self_cond_gate.weight": np.load(f"{args.weights}.gate.npy"),
        "self_cond_up.weight":   np.load(f"{args.weights}.up.npy"),
        "self_cond_down.weight": np.load(f"{args.weights}.down.npy"),
    }
    arch = None
    for f in r.fields.values():
        if f.name == "general.architecture":
            arch = bytes(f.parts[f.data[-1]]).decode()
    w = GGUFWriter(args.out, arch or "diffusion-gemma")

    # copy all KV metadata
    for key, field in r.fields.items():
        if key in ("GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"):
            continue
        try:
            w.add_key_value(field.name, field.contents(), field.types[0])
        except Exception:
            pass  # some synthetic fields can't round-trip; skip safely

    # add tensors: trained ones re-quantized to original type, others verbatim
    for t in r.tensors:
        if t.name in trained:
            new = trained[t.name].astype(np.float32)
            # re-quantize to original type via gguf quants
            try:
                qt = gguf_mod.quants.quantize(new, t.tensor_type)
            except Exception as e:
                print(f"  WARN re-quant {t.name} to {t.tensor_type.name} failed ({e}); storing F16")
                qt = new.astype(np.float16)
                w.add_tensor(t.name, qt)
                continue
            w.add_tensor(t.name, qt, raw_dtype=t.tensor_type)
            print(f"  replaced {t.name} (re-quantized to {t.tensor_type.name})")
        else:
            w.add_tensor(t.name, t.data, raw_dtype=t.tensor_type)

    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file()
    w.close()
    print(f"wrote {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--gguf", required=True); e.add_argument("--out", required=True)
    m = sub.add_parser("merge"); m.add_argument("--gguf", required=True); m.add_argument("--weights", required=True); m.add_argument("--out", required=True)
    args = ap.parse_args()
    (extract if args.cmd == "extract" else merge)(args)


if __name__ == "__main__":
    main()
