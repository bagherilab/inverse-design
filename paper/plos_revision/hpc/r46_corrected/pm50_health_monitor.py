"""Idempotent health monitor and self-healer for the pm50 reduced campaign.

Healthy running or pending arms are left untouched.  An arm whose dependency
chain has broken is repaired from its latest generation marker: the five shard
tasks are safe to replay because ``shard_runner.py`` skips completed outputs,
and a replacement prepare is chained after them.  Linear configurations that
end with ``no chains found`` are terminal by design and are never repaired.

The monitor schedules one successor through Slurm's ``singleton`` dependency
until every arm either has complete generation-4 analysis products or has
terminated with ``no chains found``.
"""

import argparse
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
from collections import defaultdict
from pathlib import Path


ANALYSIS_G4_FILES = ("all_param_df.csv", "final_metrics.csv")
WEIGHT_G4_FILES = ("weights.csv",)
BROKEN_REASON = "DependencyNeverSatisfied"
NO_CHAIN_TEXT = "no chains found"


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _run(command, check=True):
    result = subprocess.run(
        command,
        check=False,
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and result.returncode:
        raise RuntimeError(
            "command failed (%d): %s\nstdout=%s\nstderr=%s"
            % (result.returncode, " ".join(command), result.stdout, result.stderr)
        )
    return result


def _job_id(output):
    value = output.strip().split(";")[0]
    if not value.isdigit():
        raise RuntimeError("unexpected sbatch output: %r" % output)
    return value


def _field(details, name, default=""):
    match = re.search(r"(?:^| )%s=([^ ]+)" % re.escape(name), details)
    return match.group(1) if match else default


def _load_arms(campaign_root):
    arms = {}
    for config_path in sorted((campaign_root / "configs").glob("pm50_*.json")):
        config = json.loads(config_path.read_text())
        arms[config_path.stem] = {
            "config": config_path,
            "run_name": config["run_name"],
            # The linear branch uses ABCSMCRF_simplified, which computes
            # random-forest weights for propagation but does not persist a
            # weights.csv artifact. Other branches use the plain ABCSMCRF and
            # must retain that provenance file.
            "requires_weights": not (
                config.get("simplify_model") is True
                and config.get("simplify_method") == "linear"
            ),
        }
    if not arms:
        raise RuntimeError("no pm50 configs found under %s" % (campaign_root / "configs"))
    return arms


def _append_event(path, event):
    if path is None:
        return
    event = {"timestamp": _now(), **event}
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _complete(output_root, requires_weights=True):
    generation = output_root / "iter_4"
    required = ANALYSIS_G4_FILES + (WEIGHT_G4_FILES if requires_weights else ())
    return all((generation / filename).is_file() for filename in required)


def _latest_marker(campaign_root, arm):
    markers = []
    pattern = re.compile(r"generation_%s_iter_(\d+)\.json$" % re.escape(arm))
    for path in (campaign_root / "logs").glob("generation_%s_iter_*.json" % arm):
        match = pattern.search(path.name)
        if match:
            markers.append((int(match.group(1)), path))
    return max(markers, default=(None, None), key=lambda item: -1 if item[0] is None else item[0])


def _no_chain(campaign_root, config_path):
    config_pattern = re.compile(
        r"config=\S*/%s(?:\s|$)" % re.escape(config_path.name)
    )
    for path in (campaign_root / "logs").glob("prep_*.out"):
        text = path.read_text(errors="replace")
        if config_pattern.search(text) and NO_CHAIN_TEXT in text.lower():
            return True
    return False


def _arm_from_job(details, campaign_root, known_arms):
    match = re.search(r"DED_CONFIG=([^, ]+)", details)
    if not match:
        return None
    path = Path(match.group(1))
    expected = campaign_root / "configs"
    if path.resolve().parent != expected.resolve() or path.stem not in known_arms:
        return None
    return path.stem


def _partition_args(generation):
    if generation >= 4:
        return ["--partition=compute", "--account=compute-cheme"]
    return ["--partition=ckpt-all", "--account=ckpt-cheme", "--requeue"]


def _submit_prepare(campaign_root, config_path, dependency=None):
    command = [
        "sbatch",
        "--parsable",
        "--job-name=pm50hm_prep",
        "--partition=compute",
        "--account=compute-cheme",
        "--output=%s/logs/prep_%%j.out" % campaign_root,
        "--error=%s/logs/prep_%%j.err" % campaign_root,
    ]
    if dependency:
        command.append("--dependency=afterok:%s" % dependency)
    command.extend(
        [
            "--export=ALL,DED_CAMPAIGN_ROOT=%s,DED_CONFIG=%s,DED_N_SHARDS=5"
            % (campaign_root, config_path),
            str(campaign_root / "scripts" / "prepare_pm50.sbatch"),
        ]
    )
    return _job_id(_run(command).stdout)


def _submit_repair(campaign_root, config_path, marker, generation):
    command = [
        "sbatch",
        "--parsable",
        "--job-name=pm50hm_repair",
        "--output=%s/logs/health_repair_%%A_%%a.out" % campaign_root,
        "--error=%s/logs/health_repair_%%A_%%a.err" % campaign_root,
        "--array=1-5",
    ]
    command.extend(_partition_args(generation))
    command.extend(
        [
            "--export=ALL,DED_CAMPAIGN_ROOT=%s,DED_GEN_FILE=%s,DED_N_SHARDS=5,DED_CONFIG=%s"
            % (campaign_root, marker, config_path),
            str(campaign_root / "scripts" / "shard_pm50.sbatch"),
        ]
    )
    shard = _job_id(_run(command).stdout)
    prepare = _submit_prepare(campaign_root, config_path, dependency=shard)
    return shard, prepare


def _active_jobs(campaign_root, arms, events_path, repair):
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("USER is not set")
    queue = _run(["squeue", "-u", user, "-h", "-r", "-o", "%i"]).stdout
    active = defaultdict(list)
    cancelled = []
    for job in sorted(set(queue.split())):
        result = _run(["scontrol", "show", "job", "-o", job], check=False)
        if result.returncode:
            continue
        details = result.stdout.strip()
        arm = _arm_from_job(details, campaign_root, arms)
        if arm is None:
            continue
        reason = _field(details, "Reason")
        state = _field(details, "JobState")
        if reason == BROKEN_REASON:
            if repair:
                _run(["scancel", job])
                cancelled.append(job)
                _append_event(
                    events_path,
                    {
                        "action": "cancel_broken_dependency",
                        "arm": arm,
                        "job": job,
                        "state": state,
                        "reason": reason,
                    },
                )
            continue
        active[arm].append({"job": job, "state": state, "reason": reason})
    return active, cancelled


def _monitor_body(campaign_root, repair, resubmit, interval_minutes, events_path):
    arms = _load_arms(campaign_root)
    active, cancelled = _active_jobs(campaign_root, set(arms), events_path, repair)
    complete, terminal, healthy, repaired, would_repair = [], [], [], [], []
    output_base = Path("/gscratch/cheme/chiu/ARCADE_OUTPUT")

    for arm, metadata in sorted(arms.items()):
        output_root = output_base / metadata["run_name"]
        if _complete(output_root, requires_weights=metadata["requires_weights"]):
            complete.append(arm)
            continue
        if _no_chain(campaign_root, metadata["config"]):
            terminal.append(arm)
            continue
        if active.get(arm):
            healthy.append(arm)
            continue

        generation, marker = _latest_marker(campaign_root, arm)
        if not repair:
            would_repair.append({"arm": arm, "generation": generation})
            continue
        if marker is None:
            prepare = _submit_prepare(campaign_root, metadata["config"])
            record = {"arm": arm, "action": "restart_prepare", "prepare": prepare}
        else:
            generation_root = output_root / ("iter_%d" % generation)
            analysed = all(
                (generation_root / filename).is_file()
                for filename in ANALYSIS_G4_FILES
            )
            if analysed:
                prepare = _submit_prepare(campaign_root, metadata["config"])
                record = {
                    "arm": arm,
                    "action": "resume_after_analysis",
                    "generation": generation,
                    "prepare": prepare,
                }
            else:
                shard, prepare = _submit_repair(
                    campaign_root, metadata["config"], marker, generation
                )
                record = {
                    "arm": arm,
                    "action": "repair_generation",
                    "generation": generation,
                    "repair_shard": shard,
                    "repair_prepare": prepare,
                }
        repaired.append(record)
        _append_event(events_path, record)

    done = len(complete) + len(terminal)
    status = {
        "status": "complete" if done == len(arms) else "running",
        "complete_g4": len(complete),
        "terminal_no_chain": len(terminal),
        "healthy_active": len(healthy),
        "repaired": len(repaired),
        "would_repair": len(would_repair),
        "cancelled_broken": len(cancelled),
        "done": done,
        "total": len(arms),
        "terminal_arms": terminal,
        "repaired_arms": [item["arm"] for item in repaired],
        "would_repair_arms": [item["arm"] for item in would_repair],
    }
    _append_event(events_path, {"action": "health_summary", **status})

    if resubmit and status["status"] != "complete":
        wrapper = campaign_root / "scripts" / "monitor_pm50.sbatch"
        next_job = _job_id(
            _run(
                [
                    "sbatch",
                    "--parsable",
                    "--dependency=singleton",
                    "--begin=now+%dminutes" % int(interval_minutes),
                    str(wrapper),
                ]
            ).stdout
        )
        status["next_monitor"] = next_job
        _append_event(
            events_path,
            {
                "action": "schedule_next_monitor",
                "job": next_job,
                "interval_minutes": interval_minutes,
            },
        )
    print("PM50_HEALTH_MONITOR " + json.dumps(status, sort_keys=True))
    return status


def monitor_once(
    campaign_root,
    repair=True,
    resubmit=True,
    interval_minutes=30,
    dry_run=False,
):
    campaign_root = Path(campaign_root).resolve()
    if dry_run:
        return _monitor_body(
            campaign_root,
            repair=False,
            resubmit=False,
            interval_minutes=interval_minutes,
            events_path=None,
        )

    events_path = campaign_root / "health_events.jsonl"
    lock_path = campaign_root / ".health_monitor.lock"
    lock_path.touch(exist_ok=True)
    with lock_path.open("r+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "locked"}
        return _monitor_body(
            campaign_root,
            repair=repair,
            resubmit=resubmit,
            interval_minutes=interval_minutes,
            events_path=events_path,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campaign-root",
        type=Path,
        default=Path("/gscratch/cheme/chiu/bench_pm50_reduced"),
    )
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--no-resubmit", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--interval-minutes", type=int, default=30)
    args = parser.parse_args()
    if args.interval_minutes < 5:
        raise ValueError("monitor interval must be at least five minutes")
    monitor_once(
        args.campaign_root,
        repair=not args.no_repair,
        resubmit=not args.no_resubmit,
        interval_minutes=args.interval_minutes,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
