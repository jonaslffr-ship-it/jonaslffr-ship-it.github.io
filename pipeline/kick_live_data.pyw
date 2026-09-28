"""Local stand-in for the GitHub cron: the workflow's schedule has not fired once since the repo was created
(2026-09-26; push runs work, GitHub status green). Windows Task Scheduler runs this every 15 minutes on weekdays;
it dispatches live-data.yml during 09:20-16:40 ET unless a run started in the last 10 minutes, so it stays harmless
once GitHub's own schedule works. Log: %LOCALAPPDATA%\\vol-site-kick.log.
Install: schtasks /create /tn vol-site-live-data /sc weekly /d MON,TUE,WED,THU,FRI /st 14:12 /ri 15 /du 09:00
         /tr "C:\\Python314\\pythonw.exe <path>\\kick_live_data.pyw"      Remove: schtasks /delete /tn vol-site-live-data /f
ponytail: runs only while this PC is on; a cloud pinger (cron-job.org -> workflow_dispatch API) if that is not enough."""
import datetime as dt, json, os, shutil, subprocess
from zoneinfo import ZoneInfo

REPO = ["-R", "jonaslffr-ship-it/jonaslffr-ship-it.github.io"]
GH = shutil.which("gh") or r"C:\Program Files\GitHub CLI\gh.exe"


def gh(*args):
    return subprocess.run([GH, *args, *REPO], capture_output=True, text=True, creationflags=0x08000000)  # no console window


now = dt.datetime.now(ZoneInfo("America/New_York"))
if now.weekday() < 5 and dt.time(9, 20) <= now.time() <= dt.time(16, 40):
    last = json.loads(gh("run", "list", "-w", "live-data.yml", "-L", "1", "--json", "createdAt").stdout or "[]")
    age = ((dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last[0]["createdAt"].replace("Z", "+00:00"))).total_seconds() / 60
           if last else 1e9)
    msg = f"last run {age:.0f} min ago - skipped"
    if age > 10:
        r = gh("workflow", "run", "live-data.yml")
        msg = f"dispatched (exit {r.returncode}) {r.stderr.strip()[:200]}"
    with open(os.path.join(os.environ.get("LOCALAPPDATA", "."), "vol-site-kick.log"), "a", encoding="utf-8") as f:
        f.write(f"{now:%Y-%m-%d %H:%M} ET  {msg}\n")
