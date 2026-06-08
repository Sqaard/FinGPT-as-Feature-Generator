from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


ZIP_GLOB = "finportfolio_ppo_deepseek_v2_*.zip"


def find_zip(cwd: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    zips = sorted(cwd.glob(ZIP_GLOB))
    if not zips:
        raise SystemExit(f"No {ZIP_GLOB} found in the current directory.")
    if len(zips) > 1:
        names = "\n".join(p.name for p in zips)
        raise SystemExit("More than one job zip found. Use --zip ZIP_NAME.\n" + names)
    return zips[0].resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unpack, set up, and run DeepSeek v2 PPO Huawei job.")
    parser.add_argument("--zip", default="finportfolio_ppo_deepseek_v2_state_concat_robust_clipped_350k_20260523.zip", help="Job zip name/path.")
    parser.add_argument("--target-dir", default=".", help="Where to extract the package.")
    parser.add_argument("--skip-setup", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--keep-existing", action="store_true", help="Do not remove an existing extracted package root before unpacking.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("[INFO] ignoring notebook/kernel arguments:", " ".join(unknown), flush=True)

    cwd = Path.cwd().resolve()
    zip_path = find_zip(cwd, args.zip)
    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        roots = sorted({name.split("/")[0] for name in names if "/" in name})
        if len(roots) != 1:
            raise SystemExit(f"Expected exactly one package root, got: {roots}")
        package_root = target_dir / roots[0]
        if package_root.exists() and not args.keep_existing:
            print(f"[CLEAN] removing existing extracted package root: {package_root}", flush=True)
            shutil.rmtree(package_root)
        zf.extractall(target_dir)

    print(f"[OK] extracted {zip_path.name} -> {package_root}", flush=True)

    os.chdir(package_root)
    print(f"[CD] {package_root}", flush=True)
    if not args.skip_setup:
        subprocess.run([sys.executable, "setup_huawei_env.py"], check=True)
    if args.skip_run:
        return

    dependency_overlay = Path.home() / "work" / ".finportfolio_deepseek_v2_deps"
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(dependency_overlay), str(package_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    cmd = [sys.executable, "run_job.py"]
    if args.dry_run:
        cmd.append("--dry-run")
    if args.timesteps is not None:
        cmd.extend(["--timesteps", str(args.timesteps)])
    if args.force_train:
        cmd.append("--force-train")
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
