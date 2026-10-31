#!/usr/bin/env python3

import urllib.request
import os
import sys
import subprocess

HTTPLIB_VERSION = "refs/tags/v0.60.1"

vendor = {
    "https://github.com/nlohmann/json/releases/latest/download/json.hpp":     "vendor/nlohmann/json.hpp",
    "https://github.com/nlohmann/json/releases/latest/download/json_fwd.hpp": "vendor/nlohmann/json_fwd.hpp",

    "https://raw.githubusercontent.com/nothings/stb/refs/heads/master/stb_image.h": "vendor/stb/stb_image.h",

    # not using latest tag to avoid this issue: https://github.com/ggml-org/llama.cpp/pull/17179#discussion_r2515877926
    # "https://github.com/mackron/miniaudio/raw/refs/tags/0.11.24/miniaudio.h": "vendor/miniaudio/miniaudio.h",
    "https://github.com/mackron/miniaudio/raw/9634bedb5b5a2ca38c1ee7108a9358a4e233f14d/miniaudio.h": "vendor/miniaudio/miniaudio.h",

    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/httplib.h": "httplib.h",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/split.py":  "split.py",
    f"https://raw.githubusercontent.com/yhirose/cpp-httplib/{HTTPLIB_VERSION}/LICENSE":   "vendor/cpp-httplib/LICENSE",

    "https://raw.githubusercontent.com/sheredom/subprocess.h/b49c56e9fe214488493021017bf3954b91c7c1f5/subprocess.h": "vendor/sheredom/subprocess.h",
}

# local changes to vendored files, kept as diffs and applied with git apply
patch_files = [
    # backport of the fix for the stack overflow on deeply nested values (nlohmann/json#5387)
    # TODO: remove once nlohmann/json releases a version newer than 3.12.0
    "vendor/nlohmann/json-deep-nesting.patch",
]

for url, filename in vendor.items():
    print(f"downloading {url} to {filename}") # noqa: NP100
    urllib.request.urlretrieve(url, filename)

for patch_file in patch_files:
    print(f"applying {patch_file}") # noqa: NP100
    try:
        subprocess.check_call(["git", "apply", patch_file])
    except subprocess.CalledProcessError:
        print(f"Error: cannot apply {patch_file}, upstream code has changed") # noqa: NP100
        sys.exit(1)

print("Splitting httplib.h...") # noqa: NP100
try:
    subprocess.check_call([
        sys.executable, "split.py",
        "--extension", "cpp",
        "--out", "vendor/cpp-httplib"
    ])
except Exception as e:
    print(f"Error: {e}") # noqa: NP100
    sys.exit(1)
finally:
    os.remove("split.py")
    os.remove("httplib.h")
