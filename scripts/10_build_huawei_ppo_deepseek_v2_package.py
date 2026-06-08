#!/usr/bin/env python3
"""Build a Huawei Cloud package for the DeepSeek v2 PPO experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import textwrap
import zipfile
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = (
    ROOT
    / "artifacts"
    / "normalized_text_panels_deepseek_v2"
    / "processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_robust_clipped.csv"
)
DEFAULT_SCHEMA = ROOT / "feature_schema_deepseek_v2_ppo_compact.json"
DEFAULT_PACKAGE_DIR = ROOT / "artifacts" / "cloud_packages_deepseek_v2_ppo_20260523"
JOB_NAME = "finportfolio_ppo_deepseek_v2_state_concat_robust_clipped_350k"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    if dst.exists():
        shutil.rmtree(dst)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".ipynb_checkpoints")
    shutil.copytree(src, dst, ignore=ignore)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")


def build_requirements(package_root: Path) -> None:
    write_text(
        package_root / "requirements_huawei_ppo_deepseek_v2.txt",
        """
        numpy==1.26.4
        pandas==2.1.4
        PyYAML==6.0.1
        scipy==1.11.4
        gymnasium==0.29.1
        shimmy==1.3.0
        cloudpickle==3.0.0
        tqdm==4.66.4
        rich==13.7.1
        exchange-calendars==4.5.6
        pyfolio-reloaded==0.9.9
        stockstats==0.6.2
        yfinance==0.2.40
        matplotlib==3.8.4
        scikit-learn==1.4.2
        """,
    )


def build_setup_script(package_root: Path) -> None:
    write_text(
        package_root / "setup_huawei_env.py",
        r'''
        from __future__ import annotations

        import importlib
        import os
        import shutil
        import subprocess
        import sys
        from pathlib import Path


        PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
        REQ_PATH = PROJECT_ROOT / "requirements_huawei_ppo_deepseek_v2.txt"
        TARGET = Path.home() / "work" / ".finportfolio_deepseek_v2_deps"
        NO_DEPS_PACKAGES = ["stable-baselines3==2.3.2", "finrl==0.3.7"]


        def run(cmd: list[str]) -> None:
            print("[CMD]", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)


        def activate_overlay() -> None:
            target = str(TARGET)
            bad = ("modelarts-dev", "/modelarts/tools/solution/advisor", "/modelarts/tools/visualization")
            sys.path[:] = [p for p in sys.path if not any(fragment in p for fragment in bad)]
            if TARGET.exists() and target not in sys.path:
                sys.path.insert(0, target)
            importlib.invalidate_caches()


        def patch_finrl_optional_imports() -> None:
            init_path = TARGET / "finrl" / "__init__.py"
            if not init_path.exists():
                return
            marker = "Huawei PPO package patch"
            current = init_path.read_text(encoding="utf-8", errors="replace")
            if marker in current:
                return
            init_path.write_text(
                'from __future__ import annotations\n'
                '\n'
                '# Huawei PPO package patch: avoid FinRL top-level optional data-source imports.\n'
                '# This job uses precomputed panels and imports only the trading env, DRLAgent,\n'
                '# preprocessors.data_split, and plot.backtest_stats.\n',
                encoding="utf-8",
            )
            print(f"[PATCH] patched optional FinRL top-level imports in {init_path}", flush=True)


        def deps_available() -> bool:
            activate_overlay()
            patch_finrl_optional_imports()
            try:
                import exchange_calendars  # noqa: F401
                import numpy  # noqa: F401
                import pandas  # noqa: F401
                import torch  # noqa: F401
                from finrl.agents.stablebaselines3.models import DRLAgent  # noqa: F401
                from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv  # noqa: F401
                from stable_baselines3 import PPO  # noqa: F401
            except Exception as exc:
                print(f"[DEPS] missing or broken dependency: {type(exc).__name__}: {exc}", flush=True)
                return False
            print("[DEPS] training dependencies already import correctly.", flush=True)
            return True


        def write_sitecustomize() -> None:
            sitecustomize = PROJECT_ROOT / "sitecustomize.py"
            sitecustomize.write_text(
                "from pathlib import Path\n"
                "import sys\n"
                f"target = Path(r'{TARGET}')\n"
                "bad = ('modelarts-dev', '/modelarts/tools/solution/advisor', '/modelarts/tools/visualization')\n"
                "sys.path[:] = [p for p in sys.path if not any(fragment in p for fragment in bad)]\n"
                "if target.exists() and str(target) not in sys.path:\n"
                "    sys.path.insert(0, str(target))\n",
                encoding="utf-8",
            )
            print(f"[OK] sitecustomize.py written; dependency overlay: {TARGET}", flush=True)


        def main() -> None:
            os.chdir(PROJECT_ROOT)
            write_sitecustomize()
            activate_overlay()
            if os.environ.get("SKIP_INSTALL_DEPS", "0") == "1":
                print("[DEPS] SKIP_INSTALL_DEPS=1, not installing packages.", flush=True)
                return
            if deps_available() and os.environ.get("FORCE_INSTALL_DEPS", "0") != "1":
                return
            if TARGET.exists() and os.environ.get("KEEP_DEPS_OVERLAY", "0") != "1":
                print(f"[DEPS] removing stale dependency overlay: {TARGET}", flush=True)
                shutil.rmtree(TARGET)
            TARGET.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
            run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade-strategy",
                "only-if-needed",
                "--target",
                str(TARGET),
                "-r",
                str(REQ_PATH),
            ])
            run([
                sys.executable,
                "-m",
                "pip",
                "install",
                "--target",
                str(TARGET),
                "--no-deps",
                *NO_DEPS_PACKAGES,
            ])
            patch_finrl_optional_imports()
            write_sitecustomize()
            activate_overlay()
            if not deps_available():
                raise RuntimeError(
                    "Dependencies installed, but FinRL/SB3 imports still fail. "
                    "Check the messages above for the first missing package."
                )
            print(f"[OK] dependencies installed into {TARGET}", flush=True)


        if __name__ == "__main__":
            main()
        ''',
    )


def build_run_script(package_root: Path) -> None:
    write_text(
        package_root / "run_job.py",
        r'''
        from __future__ import annotations

        import argparse
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path


        PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
        DEFAULT_OUTPUT_DIR = Path("artifacts/ppo_deepseek_v2_state_concat_robust_clipped_stable_lr3e5")
        JOB_INFO = {
            "job_id": "deepseek_v2_ppo_state_concat_robust_clipped_stable_lr3e5_350k",
            "panel": "artifacts/normalized_text_panels_deepseek_v2/processed_final_fixed_external_lagclean_full_WITH_TEXT_DEEPSEEK_V2_robust_clipped.csv",
            "schema": "feature_schema_deepseek_v2_ppo_compact.json",
            "text_integration_strategy": "state_concat",
            "selected_config": "custom_custom",
            "timesteps": 350000,
            "stable_hyperparams": {
                "learning_rate": 0.00003,
                "n_steps": 1024,
                "batch_size": 256,
                "ent_coef": 0.0,
                "target_kl": 0.03,
            },
        }


        def parse_args() -> argparse.Namespace:
            parser = argparse.ArgumentParser(description="Run DeepSeek v2 PPO experiment on Huawei Cloud.")
            parser.add_argument("--timesteps", type=int, default=int(os.environ.get("TIMESTEPS", "350000")))
            parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR)))
            parser.add_argument("--selected-config", default=os.environ.get("SELECTED_CONFIG", "custom_custom"))
            parser.add_argument("--save-freq", type=int, default=int(os.environ.get("SAVE_FREQ", "25000")))
            parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "0.00003")))
            parser.add_argument("--n-steps", type=int, default=int(os.environ.get("N_STEPS", "1024")))
            parser.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "256")))
            parser.add_argument("--ent-coef", type=float, default=float(os.environ.get("ENT_COEF", "0.0")))
            parser.add_argument("--target-kl", type=float, default=float(os.environ.get("TARGET_KL", "0.03")))
            parser.add_argument("--dry-run", action="store_true")
            parser.add_argument("--force-train", action="store_true")
            parser.add_argument("--resume-from-checkpoint", default=os.environ.get("RESUME_FROM_CHECKPOINT"))
            parser.add_argument("--skip-compare", action="store_true")
            return parser.parse_args()


        def main() -> None:
            args = parse_args()
            os.chdir(PROJECT_ROOT)
            dependency_overlay = Path.home() / "work" / ".finportfolio_deepseek_v2_deps"
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            pythonpath_parts = [str(dependency_overlay), str(PROJECT_ROOT)]
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)
            env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

            cmd = [
                sys.executable,
                "scripts/03_train_backtest_ppo_with_text.py",
                "--panel",
                JOB_INFO["panel"],
                "--text-feature-schema",
                JOB_INFO["schema"],
                "--text-integration-strategy",
                JOB_INFO["text_integration_strategy"],
                "--selected-config",
                args.selected_config,
                "--output-dir",
                args.output_dir,
                "--timesteps",
                str(args.timesteps),
                "--save-freq",
                str(args.save_freq),
                "--learning-rate",
                str(args.learning_rate),
                "--n-steps",
                str(args.n_steps),
                "--batch-size",
                str(args.batch_size),
                "--ent-coef",
                str(args.ent_coef),
                "--target-kl",
                str(args.target_kl),
            ]
            if args.dry_run:
                cmd.append("--dry-run")
            if args.force_train:
                cmd.append("--force-train")
            if args.resume_from_checkpoint:
                cmd.extend(["--resume-from-checkpoint", args.resume_from_checkpoint])

            command_log = {
                "cmd": cmd,
                "job_info": JOB_INFO,
                "PYTHONPATH": env["PYTHONPATH"],
                "cwd": str(PROJECT_ROOT),
            }
            Path("cloud_job_command.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
            print("[JOB]", json.dumps(JOB_INFO, indent=2), flush=True)
            print("[PYTHONPATH]", env["PYTHONPATH"], flush=True)
            print("[CMD]", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True, env=env)

            if args.dry_run or args.skip_compare:
                return
            compare_csv = Path(args.output_dir) / "results" / "ppo_with_text_summary_for_comparison.csv"
            if not compare_csv.exists():
                print(f"[WARN] comparison CSV not found: {compare_csv}", flush=True)
                return
            compare_cmd = [
                sys.executable,
                "scripts/04_compare_with_benchmark.py",
                "--benchmark-csv",
                "ppo_without_text_BENCHMARK/benchmark_summary.csv",
                "--ppo-with-text-csv",
                str(compare_csv),
                "--output-dir",
                "artifacts/ppo_deepseek_v2_vs_benchmark",
            ]
            print("[COMPARE]", " ".join(compare_cmd), flush=True)
            subprocess.run(compare_cmd, check=True, env=env)


        if __name__ == "__main__":
            main()
        ''',
    )


def build_unpack_script(package_dir: Path, zip_name: str) -> Path:
    script_path = package_dir / "unpack_setup_run_deepseek_v2_ppo.py"
    write_text(
        script_path,
        f'''
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
                raise SystemExit(f"No {{ZIP_GLOB}} found in the current directory.")
            if len(zips) > 1:
                names = "\\n".join(p.name for p in zips)
                raise SystemExit("More than one job zip found. Use --zip ZIP_NAME.\\n" + names)
            return zips[0].resolve()


        def main() -> None:
            parser = argparse.ArgumentParser(description="Unpack, set up, and run DeepSeek v2 PPO Huawei job.")
            parser.add_argument("--zip", default="{zip_name}", help="Job zip name/path.")
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
                roots = sorted({{name.split("/")[0] for name in names if "/" in name}})
                if len(roots) != 1:
                    raise SystemExit(f"Expected exactly one package root, got: {{roots}}")
                package_root = target_dir / roots[0]
                if package_root.exists() and not args.keep_existing:
                    print(f"[CLEAN] removing existing extracted package root: {{package_root}}", flush=True)
                    shutil.rmtree(package_root)
                zf.extractall(target_dir)

            print(f"[OK] extracted {{zip_path.name}} -> {{package_root}}", flush=True)

            os.chdir(package_root)
            print(f"[CD] {{package_root}}", flush=True)
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
        ''',
    )
    return script_path


def zip_directory(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            zf.write(path, path.relative_to(source_dir.parent).as_posix())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--timesteps", type=int, default=350_000)
    args = parser.parse_args()

    package_dir = args.package_dir
    package_root = package_dir / JOB_NAME
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    copy_file(args.panel, package_root / "artifacts" / "normalized_text_panels_deepseek_v2" / args.panel.name)
    copy_file(args.schema, package_root / args.schema.name)
    copy_file(ROOT / "artifacts" / "text_features_deepseek_v2_manifest.json", package_root / "artifacts" / "text_features_deepseek_v2_manifest.json")
    copy_file(ROOT / "artifacts" / "merge_manifest_deepseek_v2.json", package_root / "artifacts" / "merge_manifest_deepseek_v2.json")
    copy_file(ROOT / "artifacts" / "normalized_text_panels_deepseek_v2" / "manifest.json", package_root / "artifacts" / "normalized_text_panels_deepseek_v2" / "manifest.json")
    copy_file(ROOT / "artifacts" / "normalized_text_panels_deepseek_v2" / "train_only_scaler_stats.json", package_root / "artifacts" / "normalized_text_panels_deepseek_v2" / "train_only_scaler_stats.json")
    copy_file(ROOT / "reports" / "drl_train_only_normalization_deepseek_v2.md", package_root / "reports" / "drl_train_only_normalization_deepseek_v2.md")
    copy_file(ROOT / "scripts" / "03_train_backtest_ppo_with_text.py", package_root / "scripts" / "03_train_backtest_ppo_with_text.py")
    copy_file(ROOT / "scripts" / "04_compare_with_benchmark.py", package_root / "scripts" / "04_compare_with_benchmark.py")
    copy_tree(ROOT / "rl_stage0_project", package_root / "rl_stage0_project")
    copy_tree(ROOT / "ppo_without_text_BENCHMARK", package_root / "ppo_without_text_BENCHMARK")

    build_requirements(package_root)
    build_setup_script(package_root)
    build_run_script(package_root)

    job_info = {
        "job_name": JOB_NAME,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "panel": "artifacts/normalized_text_panels_deepseek_v2/" + args.panel.name,
        "schema": args.schema.name,
        "strategy": "state_concat",
        "timesteps": args.timesteps,
        "command": [
            "python",
            "scripts/03_train_backtest_ppo_with_text.py",
            "--panel",
            "artifacts/normalized_text_panels_deepseek_v2/" + args.panel.name,
            "--text-feature-schema",
            args.schema.name,
            "--text-integration-strategy",
            "state_concat",
            "--output-dir",
            "artifacts/ppo_deepseek_v2_state_concat_robust_clipped",
            "--timesteps",
            str(args.timesteps),
        ],
    }
    write_text(package_root / "job_info.json", json.dumps(job_info, ensure_ascii=False, indent=2))
    write_text(
        package_root / "README_JOB.md",
        f"""
        # DeepSeek v2 PPO Huawei Job

        This package runs the FinPortfolio PPO experiment with the DeepSeek v2
        `robust_clipped` normalized text panel.

        - Panel: `artifacts/normalized_text_panels_deepseek_v2/{args.panel.name}`
        - Schema: `{args.schema.name}`
        - Strategy: `state_concat`
        - Timesteps: `{args.timesteps}`

        After training, `run_job.py` attempts to compare the result with
        `ppo_without_text_BENCHMARK/benchmark_summary.csv`.
        """,
    )

    zip_name = f"{JOB_NAME}_20260523.zip"
    zip_path = package_dir / zip_name
    zip_directory(package_root, zip_path)
    unpack_script = build_unpack_script(package_dir, zip_name)

    manifest = {
        "status": "completed",
        "package_root": str(package_root),
        "zip": str(zip_path),
        "zip_size_mb": round(zip_path.stat().st_size / 1024 / 1024, 2),
        "unpack_script": str(unpack_script),
        "huawei_command": f"python {unpack_script.name} --zip {zip_name}",
    }
    (package_dir / "package_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
