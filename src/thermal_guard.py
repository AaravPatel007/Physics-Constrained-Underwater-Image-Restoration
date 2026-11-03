"""CPU guard for this project's Python processes (no root, no system settings touched).

Every few seconds it finds every process whose working directory or command line is inside the
project, pins all of its threads to a set of logical CPUs, and reacts to the CPU temperature
(k10temp Tctl):
  normal                 -> 8 CPUs
  >= HOT for a reading   -> 3 CPUs
  still >= HOT after STOP_AFTER s on 3 CPUs, or >= CRIT -> pause (SIGSTOP) until <= RESUME, then 3 CPUs
  <= COOL for RAMP_AFTER s on 3 CPUs -> back to 6
On exit (Ctrl-C / kill) it resumes everything and restores the 8-CPU limit.
GPU work is not throttled directly; it only slows when its process is paused or starved of CPU.

  python src/thermal_guard.py            # foreground
  nohup python src/thermal_guard.py > results/logs/thermal_guard.log 2>&1 &
"""
import os
import signal
import sys
import time
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
FULL, REDUCED = set(range(8)), set(range(3))
HOT, CRIT, RESUME, COOL = 85.0, 92.0, 75.0, 78.0
STOP_AFTER, RAMP_AFTER, PERIOD = 20, 30, 2
ME = os.getpid()


def cpu_temp():
    for h in Path("/sys/class/hwmon").iterdir():
        if (h / "name").read_text().strip() == "k10temp":
            return int((h / "temp1_input").read_text()) / 1000
    return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000


def targets():
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit() or int(p.name) == ME:
            continue
        try:
            cmd = (p / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
            cwd = os.readlink(p / "cwd")
        except OSError:
            continue
        if "python" in cmd and (cwd.startswith(ROOT) or ROOT in cmd):
            out.append(int(p.name))
    return out


def pin(pids, cpus):
    for pid in pids:
        try:
            for tid in os.listdir(f"/proc/{pid}/task"):
                os.sched_setaffinity(int(tid), cpus)
        except OSError:
            pass


def send(pids, sig):
    for pid in pids:
        try:
            os.kill(pid, sig)
        except OSError:
            pass


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def main():
    state, since = "full", time.time()

    def shutdown(*_):
        pids = targets()
        send(pids, signal.SIGCONT)
        pin(pids, FULL)
        log("guard stopped: processes resumed, 8-CPU limit left in place")
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log(f"guard started: root={ROOT} hot={HOT} resume={RESUME}")
    while True:
        T, pids, now = cpu_temp(), targets(), time.time()
        prev = state
        if state == "paused":
            if T <= RESUME:
                state = "reduced"
        elif T >= CRIT or (state == "reduced" and T >= HOT and now - since >= STOP_AFTER):
            state = "paused"
        elif state == "full" and T >= HOT:
            state = "reduced"
        elif state == "reduced" and T <= COOL and now - since >= RAMP_AFTER:
            state = "full"
        if state != prev:
            since = now
            log(f"{T:.0f} C: {prev} -> {state} ({len(pids)} procs)")
            if prev == "paused":
                send(pids, signal.SIGCONT)
        if state == "paused":
            send(pids, signal.SIGSTOP)
        else:
            pin(pids, FULL if state == "full" else REDUCED)
        time.sleep(PERIOD)


if __name__ == "__main__":
    main()
