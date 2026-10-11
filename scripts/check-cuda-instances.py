#!/usr/bin/env python3
"""Checks that the CUDA flash attention vec instances the dispatch references are exactly the ones that get compiled.

The dispatch TUs (ggml/src/ggml-cuda/fattn-vec-dispatch-*.cu) call ggml_cuda_flash_attn_ext_vec_case<D, K, V> for a
list of (head size, K type, V type) combos. Each combo is instantiated in a generated file under template-instances/
and only declared extern in fattn-vec.cuh, so the dispatch TUs compile in seconds and the instances in parallel.

For the default build and for GGML_CUDA_FA_ALL_QUANTS this script checks:
- every combo the dispatch references is declared extern (no heavy implicit instantiation in a dispatch TU)
- the compiled instance files define exactly the referenced combos, each one once
  (a referenced combo that is not compiled would be an undefined symbol at link time)

usage: python3 scripts/check-cuda-instances.py           (exit code 1 on any problem)
       python3 scripts/check-cuda-instances.py --write   (rewrite the default list in fattn-vec-instances.cmake)
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CUDA = ROOT / "ggml" / "src" / "ggml-cuda"
INSTANCES = CUDA / "template-instances"
CMAKE_LIST = CUDA / "fattn-vec-instances.cmake"
MACRO_HEADERS = [CUDA / "fattn-vec.cuh", CUDA / "fattn-vec-dispatch.cuh"]

# terminal macros: their arguments are (D, type_K, type_V)
TERMINALS = ("FATTN_VEC_CASE_RET", "DECL_FATTN_VEC_CASE")
ALL_QUANTS = "GGML_CUDA_FA_ALL_QUANTS"
CMAKE_BEGIN = "# BEGIN default instances"
CMAKE_END = "# END default instances"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def join_continuations(text):
    return re.sub(r"\\\n", " ", text)


def read_macros():
    """Function-like macros of the vec instance machinery: name -> (params, body)."""
    macros = {}
    for path in MACRO_HEADERS:
        text = join_continuations(strip_comments(path.read_text()))
        for m in re.finditer(r"^[ \t]*#[ \t]*define[ \t]+(\w+)\(([^)]*)\)(.*)$", text, flags=re.M):
            name, params, body = m.group(1), [p.strip() for p in m.group(2).split(",")], m.group(3)
            if name in TERMINALS or "FATTN_VEC" not in name:
                continue
            macros[name] = (params, body)
    return macros


def expand(text, macros, extern_only=False):
    """All (D, type_K, type_V) combos that the macro invocations in text expand to."""
    names = "|".join(sorted(list(macros) + list(TERMINALS), key=len, reverse=True))
    combos = []
    for m in re.finditer(r"(\bextern\s+)?\b(" + names + r")\s*\(([^()]*)\)", text):
        is_extern, name, args = m.group(1) is not None, m.group(2), [a.strip() for a in m.group(3).split(",")]
        if name in TERMINALS:
            if extern_only is None or is_extern == extern_only:
                combos.append((int(args[0]), args[1], args[2]))
            continue
        params, body = macros[name]
        if len(params) != len(args):
            raise SystemExit(f"cannot expand {name}({', '.join(args)})")
        for p, a in zip(params, args):
            body = re.sub(r"\b" + re.escape(p) + r"\b", a, body)
        combos += expand(body, macros, extern_only=None if is_extern else extern_only)
    return combos


def active_text(path, all_quants):
    """The source text of path as the compiler sees it for one value of GGML_CUDA_FA_ALL_QUANTS, without #defines.

    Only GGML_CUDA_FA_ALL_QUANTS is evaluated; the branches of any other conditional are all kept (conservative)."""
    lines = join_continuations(strip_comments(path.read_text())).split("\n")
    stack = []  # per open conditional: True/False for an evaluated branch state, None if not evaluated
    out = []
    for line in lines:
        m = re.match(r"\s*#\s*(\w+)\s*(.*)", line)
        if m:
            directive, arg = m.group(1), m.group(2).strip()
            if directive in ("if", "ifdef", "ifndef"):
                if directive == "ifdef" and arg == ALL_QUANTS:
                    stack.append(all_quants)
                elif directive == "ifndef" and arg == ALL_QUANTS:
                    stack.append(not all_quants)
                elif ALL_QUANTS in arg:
                    raise SystemExit(f"{path.name}: unsupported condition on {ALL_QUANTS}: {line.strip()}")
                else:
                    stack.append(None)
            elif directive in ("elif", "elifdef", "elifndef"):
                if stack[-1] is not None:
                    raise SystemExit(f"{path.name}: unsupported #elif on {ALL_QUANTS}")
            elif directive == "else":
                if stack[-1] is not None:
                    stack[-1] = not stack[-1]
            elif directive == "endif":
                stack.pop()
            continue
        if all(s is not False for s in stack):
            out.append(line)
    return "\n".join(out)


def referenced(macros, all_quants):
    """Combos the dispatch TUs reference: {combo: [files]}."""
    refs = {}
    for path in sorted(CUDA.glob("*.cu")):
        text = active_text(path, all_quants)
        for combo in expand(text, macros, extern_only=False):
            refs.setdefault(combo, []).append(path.name)
        for m in re.finditer(r"ggml_cuda_flash_attn_ext_vec_case\s*<", text):
            raise SystemExit(f"{path.name}: direct ggml_cuda_flash_attn_ext_vec_case<...> call, use the dispatch macros")
    return refs


def extern_declared(macros):
    text = join_continuations(strip_comments((CUDA / "fattn-vec.cuh").read_text()))
    text = re.sub(r"^[ \t]*#[ \t]*define[^\n]*", "", text, flags=re.M)
    return set(expand(text, macros, extern_only=True))


def defined_in(path, macros):
    text = join_continuations(strip_comments(path.read_text()))
    text = re.sub(r"^[ \t]*#[ \t]*define[^\n]*", "", text, flags=re.M)
    return expand(text, macros, extern_only=False)


def default_list():
    text = CMAKE_LIST.read_text()
    block = text[text.index(CMAKE_BEGIN):text.index(CMAKE_END)]
    return re.findall(r"fattn-vec-instance-[\w-]+\.cu", block)


def compiled_files(all_quants):
    if all_quants:
        return sorted(INSTANCES.glob("fattn-vec*.cu"))
    return [INSTANCES / name for name in default_list()]


def short(t):
    return t.replace("GGML_TYPE_", "").lower()


def instance_file(combo):
    d, k, v = combo
    return f"fattn-vec-instance-{short(k)}-{short(v)}{'-d512' if d == 512 else ''}.cu"


def write_default_list(macros):
    names = sorted({instance_file(c) for c in referenced(macros, all_quants=False)})
    body = "".join(f"    {n}\n" for n in names)
    text = CMAKE_LIST.read_text()
    head, rest = text.split(CMAKE_BEGIN, 1)
    tail = rest.split(CMAKE_END, 1)[1]
    first_line = rest.split("\n", 1)[0]
    CMAKE_LIST.write_text(f"{head}{CMAKE_BEGIN}{first_line}\nset(GGML_CUDA_FATTN_VEC_DEFAULT_INSTANCES\n{body})\n{CMAKE_END}{tail}")


def fmt(combo):
    return f"D={combo[0]} K={short(combo[1])} V={short(combo[2])}"


def check(macros, all_quants):
    mode = ALL_QUANTS if all_quants else "default"
    problems = []
    refs = referenced(macros, all_quants)
    externs = extern_declared(macros)

    for combo in sorted(set(refs) - externs):
        problems.append(f"{mode}: {fmt(combo)} is referenced in {', '.join(refs[combo])} but not declared extern "
                        "(it would be instantiated in the dispatch TU)")

    compiled = {}
    files = compiled_files(all_quants)
    for path in files:
        if not path.exists():
            problems.append(f"{mode}: {path.name} is listed but does not exist")
            continue
        for combo in defined_in(path, macros):
            compiled.setdefault(combo, []).append(path.name)
    for combo, where in sorted(compiled.items()):
        if len(where) > 1:
            problems.append(f"{mode}: {fmt(combo)} is defined in more than one file: {', '.join(where)}")
    for combo in sorted(set(refs) - set(compiled)):
        problems.append(f"{mode}: {fmt(combo)} is referenced in {', '.join(refs[combo])} but not compiled")
    for combo in sorted(set(compiled) - set(refs)):
        problems.append(f"{mode}: {fmt(combo)} is compiled in {', '.join(compiled[combo])} but never referenced")

    print(f"{mode}: {len(files)} instance files, {len(compiled)} instances compiled, {len(refs)} referenced, "
          f"{len(set(refs) - externs)} instantiated in dispatch TUs")
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="rewrite the default list in fattn-vec-instances.cmake")
    args = parser.parse_args()

    macros = read_macros()
    if args.write:
        write_default_list(macros)

    problems = check(macros, all_quants=False) + check(macros, all_quants=True)
    for line in problems:
        print(line)
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
