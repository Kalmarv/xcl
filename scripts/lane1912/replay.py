"""Collect random-input readbacks for #1912, one process and one row per seed."""
import argparse
from collections import Counter
import csv
from hashlib import sha256
import os
from pathlib import Path
import subprocess
import sys

from probe import probe

HEAD = "5d0c1f0b8a802712e9542f0ffa269d9e36b6747f"
FIELDS = (
    "tag hash seed profile compared_words hlsl_changed lane_changed differing_words "
    "flat_word resource resource_word hlsl_word lane_word recheck tiles descriptors probe_diff"
).split()


def compile_shader(dxc, source, timeout):
    size = source.stat().st_size
    flags = ["-O0"] if size > 8 * 1024**2 else ["-O1"] if size > 1024**2 else []
    subprocess.run(
        [dxc, "-E", "main", "-T", "cs_6_6", "-HV", "2021", *flags, source, "-Fo", source.with_suffix(".dxil")],
        check=True, timeout=timeout,
    )


def compare(args, row):
    root = args.out
    h = row["hash"]
    mode = "split" if args.split and row["wave"] == "64" else "native"
    lane = root / "lane" / (h + ".dxil")
    ref = root / "ref" / (h + ".hlsl")
    if not lane.exists() or not ref.exists():
        return {}, {}, row["split_refusal"] or "missing translation; see export.log"
    for path, key in [(lane, mode + "_lane_sha256"), (ref, mode + "_hlsl_sha256")]:
        if sha256(path.read_bytes()).hexdigest() != row[key]:
            raise ValueError(f"unexpected {path.suffix} digest for {h} at {HEAD[:8]}")
    diagnostic = root / "probe" / ref.name
    diagnostic.write_text(probe(ref.read_text(encoding="utf8")), encoding="utf8")
    compile_shader(args.dxc, ref, args.timeout)
    compile_shader(args.dxc, diagnostic, args.timeout)
    readbacks = {}
    skipped = {}
    details = {}
    logfile = root / "rows" / (h + ".log")
    try:
        with logfile.open("w", encoding="utf8") as log:
            result = subprocess.run(
                [args.bin_dir / "lane1912_replay.exe", root, h, *args.seeds],
                stdout=log, stderr=log, timeout=args.timeout,
            )
        failure = f"runner exit {result.returncode}" if result.returncode else ""
    except subprocess.TimeoutExpired:
        failure = f"runner timeout after {args.timeout}s"
    for line in logfile.read_text(errors="replace").splitlines():
        parts = next(csv.reader([line]))
        if parts[:1] == ["ROW"]:
            if len(parts) != len(FIELDS) or parts[1] != h or parts[2] in readbacks or parts[2] not in args.seeds:
                raise ValueError("malformed or duplicate GPU row")
            readbacks[parts[2]] = dict(zip(FIELDS, parts))
        elif parts[:1] == ["DETAIL"] and len(parts) == 6 and parts[1] == h:
            details[parts[2]] = dict(zip(["signed_zero_words", "both_nan_words", "other_words"], parts[3:]))
        elif parts[:1] == ["SKIP"] and len(parts) >= 4:
            skipped[parts[2]] = ",".join(parts[3:])
        elif line.startswith("Error:"):
            failure = line
    for seed, values in readbacks.items():
        if seed in details:
            if sum(map(int, details[seed].values())) != int(values["differing_words"]):
                raise ValueError("inconsistent mismatch classification")
            values.update(details[seed])
    return readbacks, skipped, failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["pack", "dxc", "out", "bin-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", action="store_true", help="split wave64 programs into two wave32 halves")
    parser.add_argument("--wave", choices=["32", "64"], help="select programs by native guest wave size")
    parser.add_argument("--hash", help="select one manifest entry")
    parser.add_argument("--seeds", nargs="+", default=["191201", "191203", "191211"])
    parser.add_argument("--timeout", type=int, default=90, help="seconds per DXC invocation or GPU process")
    args = parser.parse_args()
    if args.timeout <= 0 or len(set(args.seeds)) != len(args.seeds) or any(not s.isdecimal() or int(s) >= 2**64 - 2 for s in args.seeds):
        parser.error("use a positive timeout and distinct unsigned 64-bit seeds below 2^64-2")
    with Path(__file__).with_name("admissions.csv").open(newline="") as src:
        rows = list(csv.DictReader(src))
    if len(rows) != 111 or len({row["hash"] for row in rows}) != 111 or any(row["head"] != HEAD for row in rows):
        parser.error("missing or inconsistent pinned admission manifest")
    rows = [row for row in rows if (not args.wave or args.wave == row["wave"]) and (not args.hash or args.hash.lower() == row["hash"])]
    if not rows:
        parser.error("no matching programs")
    root = args.out.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error("choose an empty output directory to keep revisions, modes and runs separate")
    for form in ("ref", "lane", "probe", "rows"):
        (root / form).mkdir(exist_ok=True, parents=True)
    selection = root / "selection.csv"
    with selection.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=["hash", "fssm", "program", "source_bytes"], extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    env = os.environ.copy()
    env.pop("LANE1912_SPLIT", None)
    if args.split:
        env["LANE1912_SPLIT"] = "1"
    with (root / "export.log").open("w", encoding="utf8") as log:
        subprocess.run([args.bin_dir / "lane1912_split.exe", args.pack.resolve(), selection, root], env=env, stdout=log, check=True)
    totals = Counter()
    with (root / "results.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=["head", "form", "status", "reason", *FIELDS[1:], "signed_zero_words", "both_nan_words", "other_words"])
        writer.writeheader()
        for row in rows:
            try:
                results, skipped, failure = compare(args, row)
            except (subprocess.SubprocessError, OSError, ValueError) as exc:
                results, skipped, failure = {}, {}, str(exc)
            for seed in args.seeds:
                values = results.get(seed, {})
                reason = skipped.get(seed, failure)
                if not values:
                    status = "unavailable"
                elif int(values["differing_words"]):
                    status = "mismatch"
                elif int(values["probe_diff"]):
                    status = "probe drift"
                elif int(values["hlsl_changed"]):
                    status = "equal, changed output"
                else:
                    status = "equal, unchanged output"
                form = "split64" if args.split and row["wave"] == "64" else "native" + row["wave"]
                writer.writerow({"head": HEAD, "form": form, "status": status, "reason": reason, "hash": row["hash"], "seed": seed,
                                 **{k: v for k, v in values.items() if k != "tag"}})
                totals[status] += 1
            output.flush()
            print(row["hash"], flush=True)
    print(dict(totals))
    return 1 if totals["mismatch"] else 2 if totals["unavailable"] or totals["probe drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
