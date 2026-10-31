#!/usr/bin/env python3
"""
hw7_tool.py - helper CLI to rebuild and check every HW7 submission file.
Run it from HW7_files/ (the folder with train_SVHN.py; nanoGPT/ lives inside).
Do NOT submit this file.

  python hw7_tool.py results [--only 1 2 7] [--force]   # 7 LeNet configs -> results.txt
  python hw7_tool.py augmentation [--aug affine]        # -> augmentation_analysis.txt
  python hw7_tool.py nanogpt                            # -> generated_nanogpt.txt
  python hw7_tool.py distil [--temperature 0.7 ...]     # patch get_params + regenerate
  python hw7_tool.py check                              # validate all files
  python hw7_tool.py package                            # copy the 6 files to ./submission/
"""
import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "results_cache.json")
NANO_DIR = os.path.join(ROOT, "nanoGPT")

SUBMIT_FILES = [
    "student_code.py", "results.txt", "augmentation_analysis.txt",
    "generated_nanogpt.txt", "generated_distilgpt.txt", "distilgpt2_sft_cpu.py",
]

# (name, extra args for train_SVHN.py) in the order results.txt needs
CONFIGS = [
    ("default", []),
    ("bs8", ["--batch-size", "8"]),
    ("bs16", ["--batch-size", "16"]),
    ("lr0.05", ["--lr", "0.05"]),
    ("lr0.01", ["--lr", "0.01"]),
    ("epochs20", ["--epochs", "20"]),
    ("epochs5", ["--epochs", "5"]),
]

# (temperature, top_k, max_new_tokens, seed) for nanoGPT samples
NANO_RUNS = [(0.8, 200, 300, 1337), (0.3, 10, 300, 1), (1.2, 200, 300, 2)]
NANO_NOISE = ("Overriding", "number of parameters", "Loading meta")


# ---------------------------------------------------------------- helpers
def run(cmd, cwd=ROOT, capture=False):
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd, text=True, capture_output=capture)
    if p.returncode != 0:
        if capture:
            print(p.stdout, p.stderr)
        sys.exit(f"Command failed with code {p.returncode}")
    return p.stdout if capture else ""


def load_cache():
    if os.path.exists(CACHE):
        with open(CACHE) as f:
            return json.load(f)
    return {}


def save_cache(c):
    with open(CACHE, "w") as f:
        json.dump(c, f, indent=2)


def train_and_eval(name, extra):
    """Fresh train run, then eval model_best. Returns accuracy in percent."""
    out = os.path.join(ROOT, "outputs")
    if os.path.exists(out):  # never resume or overwrite an old run
        stale = f"outputs_stale_{int(time.time())}"
        print(f"Moving existing outputs/ to {stale}/")
        shutil.move(out, os.path.join(ROOT, stale))
    run([PY, "train_SVHN.py", *extra])
    text = run([PY, "eval_SVHN.py", "--load", "./outputs/model_best.pth.tar"],
               capture=True)
    m = re.findall(r"Accuracy:\s*([\d.]+)%", text)
    if not m:
        sys.exit("Could not parse accuracy from eval output:\n" + text)
    acc = float(m[-1])
    runs = os.path.join(ROOT, "runs")
    os.makedirs(runs, exist_ok=True)
    dest = os.path.join(runs, name)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    shutil.move(out, dest)
    return acc


# ---------------------------------------------------------------- commands
def cmd_results(args):
    cache = load_cache()
    chosen = set(args.only) if args.only else set(range(1, len(CONFIGS) + 1))
    for i, (name, extra) in enumerate(CONFIGS, start=1):
        if i not in chosen:
            continue
        if name in cache and not args.force:
            print(f"[{i}] {name}: cached {cache[name]:.2f} (use --force to redo)")
            continue
        print(f"\n=== Config {i}/{len(CONFIGS)}: {name} ===")
        cache[name] = train_and_eval(name, extra)
        save_cache(cache)  # save after every run so you can stop and resume
        print(f"[{i}] {name}: {cache[name]:.2f}")
    missing = [n for n, _ in CONFIGS if n not in cache]
    if missing:
        print("\nresults.txt NOT written yet. Missing:", ", ".join(missing))
        return
    with open(os.path.join(ROOT, "results.txt"), "w") as f:
        f.write("\n".join(f"{cache[n]:.2f}" for n, _ in CONFIGS) + "\n")
    print("\nWrote results.txt")


def aug_text(base, aug, aug_name):
    d = aug - base
    if d >= 0.5:
        verdict = ("Data augmentation helped modestly, though with a single run "
                   "per setting some of this gap could be due to random variation.")
        change = "an improvement"
    elif d <= -0.5:
        verdict = ("Data augmentation did not help here, which may be because "
                   "10 epochs is short and augmented data is harder for a small "
                   "model like LeNet-5 to fit.")
        change = "a decrease"
    else:
        verdict = ("Data augmentation made little difference, and with a single "
                   "run per setting the gap is within what random variation "
                   "could explain.")
        change = "a change"
    desc = {"affine": "RandomAffine (translations of up to 10%)",
            "colorjitter": "ColorJitter (brightness, contrast, saturation 0.1)",
            "rotation": "an extra RandomRotation(10)"}[aug_name]
    return " ".join([
        f"The baseline LeNet-5 reached {base:.2f}% validation accuracy, while "
        f"training with {desc} reached {aug:.2f}%, {change} of {abs(d):.2f} "
        f"percentage points.",
        verdict,
        "Augmentation can help CNNs generalize because each epoch shows the "
        "model slightly different versions of the same digit, which reduces "
        "overfitting to exact pixel positions and encourages features that are "
        "robust to small variations.",
        "This fits SVHN well, since house-number digits are often off-center, "
        "tilted, or differently lit in real photos, so random transformations "
        "mimic the variation seen at test time.",
        "Note that the default pipeline already applies a small random "
        "rotation, so the baseline was not completely augmentation-free.",
    ]) + "\n"


def cmd_augmentation(args):
    cache = load_cache()
    if "default" not in cache:
        sys.exit("Baseline missing. Run: python hw7_tool.py results --only 1")
    key = f"aug_{args.aug}"
    if key not in cache or args.force:
        cache[key] = train_and_eval(key, ["--augmentation", args.aug])
        save_cache(cache)
    text = aug_text(cache["default"], cache[key], args.aug)
    with open(os.path.join(ROOT, "augmentation_analysis.txt"), "w") as f:
        f.write(text)
    print(text)
    print("Wrote augmentation_analysis.txt")


def cmd_nanogpt(args):
    out_path = os.path.join(ROOT, "generated_nanogpt.txt")
    chunks = []
    for temp, top_k, n_tok, seed in NANO_RUNS:
        text = run([PY, "sample.py", "--out_dir=out-tale-of-two-cities",
                    "--device=cpu", f"--temperature={temp}", f"--top_k={top_k}",
                    f"--max_new_tokens={n_tok}", "--num_samples=2",
                    f"--seed={seed}"], cwd=NANO_DIR, capture=True)
        lines = [l for l in text.splitlines() if not l.startswith(NANO_NOISE)]
        body = "\n".join(lines).strip("\n")
        chunks.append(f"=== temperature={temp}, top_k={top_k}, "
                      f"max_new_tokens={n_tok}, seed={seed} ===\n{body}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chunks))
    print("Wrote generated_nanogpt.txt")


def patch_get_params(path, values):
    """Edit only the real params dict (the last 'params = {') in get_params."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    start = src.rfind("params = {")
    end = src.find("}", start)
    if start == -1 or end == -1:
        sys.exit("Could not find the params dict in " + path)
    block = src[start:end]
    for k, v in values.items():
        block, n = re.subn(rf'("{k}"\s*:\s*)[\d.]+', rf"\g<1>{v}", block)
        if n != 1:
            sys.exit(f"Could not patch key {k}")
    shutil.copy(path, path + ".bak")
    with open(path, "w", encoding="utf-8") as f:
        f.write(src[:start] + block + src[end:])


def cmd_distil(args):
    script = os.path.join(ROOT, "distilgpt2_sft_cpu.py")
    values = {"temperature": args.temperature, "top_p": args.top_p,
              "max_new_tokens": args.max_new_tokens,
              "repetition_penalty": args.repetition_penalty}
    patch_get_params(script, values)
    print("Patched get_params() ->", values, "(backup: distilgpt2_sft_cpu.py.bak)")
    if args.train:
        run([PY, "make_data_csv.py"])
        run([PY, "distilgpt2_sft_cpu.py", "--data", "data.csv", "--mode", "train"])
    elif not os.path.isdir(os.path.join(ROOT, "distilgpt2_sft_cpu")):
        sys.exit("No distilgpt2_sft_cpu/ folder. Re-run with --train.")
    # output comes from the real submitted script, so file and code match
    run([PY, "distilgpt2_sft_cpu.py", "--mode", "gen"])
    print("Wrote generated_distilgpt.txt")


def cmd_check(args):
    ok = True

    def report(good, msg):
        nonlocal ok
        ok = ok and good
        print(("PASS  " if good else "FAIL  ") + msg)

    def read(name):
        p = os.path.join(ROOT, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            report(False, f"{name} missing or empty")
            return None
        return open(p, encoding="utf-8").read()

    # student_code.py: nothing at top level except imports/defs
    src = read("student_code.py")
    if src:
        tree = ast.parse(src)
        bad = [n for n in tree.body if not isinstance(
            n, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
            and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
        report(not bad, "student_code.py has no code outside definitions")
        prints = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and getattr(n.func, "id", "") == "print"]
        print(f"INFO  student_code.py has {len(prints)} print() call(s); "
              "the skeleton's own epoch prints are fine, remove your debug ones")

    t = read("results.txt")
    if t:
        vals = t.split()
        report(len(vals) == 7 and all(re.fullmatch(r"\d+(\.\d+)?", v) for v in vals),
               "results.txt has 7 numeric lines")

    t = read("augmentation_analysis.txt")
    if t:
        n = len([s for s in re.split(r"(?<=[.!?])\s+", t.strip()) if s])
        report(3 <= n <= 5, f"augmentation_analysis.txt has {n} sentences (need 3-5)")

    t = read("generated_nanogpt.txt")
    if t:
        report(not any(l.startswith(NANO_NOISE) for l in t.splitlines()),
               "generated_nanogpt.txt has no log lines")
        n = t.count("=== temperature=")
        report(n >= 2, f"generated_nanogpt.txt has {n} labeled settings")

    t = read("generated_distilgpt.txt")
    if t:
        n = len(re.findall(r"^=== Prompt \d \(Sample \d\) ===$", t, re.M))
        report(n == 9, f"generated_distilgpt.txt has {n}/9 sections")
        parts = re.split(r"=== Generation ===\n", t)[1:]
        report(len(parts) == 9 and all(
            len(re.sub(r"=== Prompt.*", "", p).strip()) > 20 for p in parts),
            "every distilgpt sample is non-empty")

    src = read("distilgpt2_sft_cpu.py")
    if src:
        report("raise NotImplementedError" not in src,
               "distilgpt2_sft_cpu.py has no leftover NotImplementedError")
        report("raise params" not in src, "get_params() uses return, not raise")

    print("\nALL GOOD" if ok else "\nFix the FAIL items above.")
    return ok


def cmd_package(args):
    if not cmd_check(args):
        sys.exit("Not packaging until checks pass.")
    dest = os.path.join(ROOT, "submission")
    os.makedirs(dest, exist_ok=True)
    for f in SUBMIT_FILES:
        shutil.copy(os.path.join(ROOT, f), os.path.join(dest, f))
    print(f"\nCopied {len(SUBMIT_FILES)} files to {dest}/ - upload them to Gradescope.")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("results", help="run the 7 LeNet configs")
    p.add_argument("--only", type=int, nargs="+", help="config numbers 1-7")
    p.add_argument("--force", action="store_true", help="redo cached configs")
    p.set_defaults(fn=cmd_results)

    p = sub.add_parser("augmentation", help="run augmented training + write analysis")
    p.add_argument("--aug", choices=["affine", "colorjitter", "rotation"],
                   default="affine")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_augmentation)

    p = sub.add_parser("nanogpt", help="regenerate generated_nanogpt.txt")
    p.set_defaults(fn=cmd_nanogpt)

    p = sub.add_parser("distil", help="patch get_params and regenerate")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--repetition_penalty", type=float, default=1.2)
    p.add_argument("--train", action="store_true", help="also rebuild data + retrain")
    p.set_defaults(fn=cmd_distil)

    p = sub.add_parser("check", help="validate all submission files")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("package", help="check, then copy files to ./submission/")
    p.set_defaults(fn=cmd_package)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()