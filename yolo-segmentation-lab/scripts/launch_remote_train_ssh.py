#!/usr/bin/env python3
import argparse
import subprocess
import sys


def _ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def build_remote_ps_command(args) -> str:
    out_log = args.stdout_log or r"runs\segment\remote_train_stdout.log"
    err_log = args.stderr_log or r"runs\segment\remote_train_stderr.log"
    arg_list = [
        "scripts/train_yolo_seg.py",
        "--model", args.model,
        "--epochs", str(args.epochs),
        "--imgsz", str(args.imgsz),
        "--batch", str(args.batch),
        "--device", str(args.device),
        "--workers", str(args.workers),
    ]
    ps_args = ", ".join(_ps_quote(a) for a in arg_list)
    repo = _ps_quote(args.remote_repo)
    py = _ps_quote(args.remote_python)
    out_rel = _ps_quote(out_log)
    err_rel = _ps_quote(err_log)
    return (
        f"$repo={repo}; "
        f"$py={py}; "
        f"$outRel={out_rel}; "
        f"$errRel={err_rel}; "
        "$runDir=Join-Path $repo 'runs\\segment'; "
        "New-Item -ItemType Directory -Force -Path $runDir | Out-Null; "
        "$outLog=Join-Path $repo $outRel; "
        "$errLog=Join-Path $repo $errRel; "
        f"$trainArgs=@({ps_args}); "
        "Start-Process -FilePath $py -WorkingDirectory $repo -ArgumentList $trainArgs "
        "-RedirectStandardOutput $outLog -RedirectStandardError $errLog -WindowStyle Hidden | Out-Null; "
        "Write-Output ('Remote training started. stdout=' + $outLog + ' stderr=' + $errLog)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="SSH target, e.g. user@training-pc")
    ap.add_argument("--remote-repo", required=True, help="Repo path on the remote machine")
    ap.add_argument("--remote-python", required=True, help="Python executable on the remote machine")
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--imgsz", type=int, required=True)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--stdout-log", default="")
    ap.add_argument("--stderr-log", default="")
    ap.add_argument("--print-command", action="store_true")
    args = ap.parse_args()

    remote_ps = build_remote_ps_command(args)
    ssh_cmd = [
        "ssh",
        args.target,
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        remote_ps,
    ]

    if args.print_command:
        print(" ".join(ssh_cmd))
        return

    try:
        completed = subprocess.run(ssh_cmd, text=True, capture_output=True)
    except FileNotFoundError:
        print("Local SSH client not found. Install OpenSSH client on this PC.", file=sys.stderr)
        raise SystemExit(1)

    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.returncode != 0:
        if completed.stderr.strip():
            print(completed.stderr.strip(), file=sys.stderr)
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
