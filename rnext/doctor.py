"""Environment doctor: local checks for the music-restoration setup.

Hard checks (any failure -> exit 1), each failure carries an install command
per package family (apt/dnf/pacman/brew):
  - python >= 3.10;
  - importable numpy, scipy, matplotlib, mutagen, PIL, bs4, requests,
    sklearn, joblib;
  - external binaries: ffmpeg, ffprobe, flac, metaflac, shntool, cuetag;
  - config/workspace.json exists and is valid (dict with data_root,
    budget_bytes, dirs -- same keys rnext.workspace.load_config reads);
  - data_root exists and is writable;
  - >= 1 GiB free on data_root.

Warn checks (exit 2 when nothing hard failed):
  - rutracker cookies file is missing;
  - transmission RPC does not answer;
  - gio is missing (MTP delivery);
  - adb is missing (ADB delivery).

Every probe is local: shutil.which, the workspace config, a write probe and
shutil.disk_usage on data_root, and a TCP connect to transmission on
127.0.0.1 (loopback only). No network requests are made.

CLI usage (thin wrappers; the logic lives here):
    python3 scripts/doctor.py [--json] [--config PATH]
    python3 -m rnext.doctor [--json] [--config PATH]

Output: a status/check/fix table, or with --json a report
    {"python": ..., "platform": ..., "exit_code": 0|1|2,
     "checks": [{"id", "name", "severity", "status", "detail", "fix",
                 "fix_by_platform"?}, ...]}
where severity is "hard"|"warn" and status is "ok"|"fail"|"warn"
(hard checks never report "warn", warn checks never "fail").
Exit codes: 0 all green; 1 at least one hard failure; 2 only warn failures.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import socket
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "workspace.json"

MIN_PYTHON = (3, 10)
MIN_FREE_BYTES = 1024 ** 3  # 1 GiB

PY_MODULES = (
    "numpy", "scipy", "matplotlib", "mutagen", "PIL",
    "bs4", "requests", "sklearn", "joblib",
)
TOOLS = ("ffmpeg", "ffprobe", "flac", "metaflac", "shntool", "cuetag")

TRANSMISSION_HOST = "127.0.0.1"
TRANSMISSION_PORT = 9091

SECRETS_DIR = Path.home() / ".config" / "music-restoration" / "secrets"
SECRETS_NAME = "rutracker_cookies.txt"

FAMILIES = ("apt", "dnf", "pacman", "brew")

FAMILY_INSTALL = {
    "apt": "sudo apt install {}",
    "dnf": "sudo dnf install {}",
    "pacman": "sudo pacman -S {}",
    "brew": "brew install {}",
}

# tool -> (apt, dnf, pacman, brew) package; ffprobe/metaflac ship with
# ffmpeg/flac respectively.
TOOL_PACKAGES = {
    "ffmpeg": ("ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg"),
    "ffprobe": ("ffmpeg", "ffmpeg", "ffmpeg", "ffmpeg"),
    "flac": ("flac", "flac", "flac", "flac"),
    "metaflac": ("flac", "flac", "flac", "flac"),
    "shntool": ("shntool", "shntool", "shntool", "shntool"),
    "cuetag": ("cuetools", "cuetools", "cuetools", "cuetools"),
}

# module -> (apt, dnf, pacman); brew has no reliable formulas for these,
# pip is the brew-column answer.
MODULE_PACKAGES = {
    "numpy": ("python3-numpy", "python3-numpy", "python-numpy"),
    "scipy": ("python3-scipy", "python3-scipy", "python-scipy"),
    "matplotlib": ("python3-matplotlib", "python3-matplotlib", "python-matplotlib"),
    "mutagen": ("python3-mutagen", "python3-mutagen", "python-mutagen"),
    "PIL": ("python3-pil", "python3-pillow", "python-pillow"),
    "bs4": ("python3-bs4", "python3-beautifulsoup4", "python-beautifulsoup4"),
    "requests": ("python3-requests", "python3-requests", "python-requests"),
    "sklearn": ("python3-sklearn", "python3-scikit-learn", "python-scikit-learn"),
    "joblib": ("python3-joblib", "python3-joblib", "python-joblib"),
}

PYTHON_FIX = {
    "apt": "sudo apt install python3.12",
    "dnf": "sudo dnf install python3.12",
    "pacman": "sudo pacman -S python",
    "brew": "brew install python@3.12",
}

GIO_FIX = {
    "apt": "sudo apt install libglib2.0-bin",
    "dnf": "sudo dnf install glib2",
    "pacman": "sudo pacman -S glib2",
    "brew": "brew install glib",
}

ADB_FIX = {
    "apt": "sudo apt install adb",
    "dnf": "sudo dnf install android-tools",
    "pacman": "sudo pacman -S android-tools",
    "brew": "brew install android-platform-tools",
}

CONFIG_FIX = "cp config/workspace.example.json config/workspace.json и задать data_root"
DATA_ROOT_FIX = (
    "создать каталог data_root и дать права записи (chmod u+w), "
    "либо поправить data_root в config/workspace.json"
)
DISK_FIX = "освободить не менее 1 GiB на data_root либо указать другой data_root"


def python_version() -> str:
    return ".".join(str(part) for part in sys.version_info[:3])


def pkg_family() -> str:
    """Package-manager family: apt | dnf | pacman | brew | unknown."""
    if sys.platform == "darwin":
        return "brew"
    if not sys.platform.startswith("linux"):
        return "unknown"
    info = {}
    try:
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            key, _, val = line.partition("=")
            info[key.strip()] = val.strip().strip('"')
    except OSError:
        return "unknown"
    ids = f"{info.get('ID', '')} {info.get('ID_LIKE', '')}".lower()
    if any(x in ids for x in ("debian", "ubuntu", "mint", "kali")):
        return "apt"
    if any(x in ids for x in ("fedora", "rhel", "centos", "rocky", "alma", "suse")):
        return "dnf"
    if any(x in ids for x in ("arch", "manjaro", "endeavouros")):
        return "pacman"
    return "unknown"


def install_cmd(family: str, packages: list[str]) -> str:
    if family in FAMILY_INSTALL:
        return FAMILY_INSTALL[family].format(" ".join(packages))
    return "установите вручную: " + ", ".join(packages)


def _rec(cid: str, name: str, severity: str, status: str, detail: str,
         fix: str = "", fix_by_platform: dict | None = None) -> dict:
    rec = {
        "id": cid,
        "name": name,
        "severity": severity,
        "status": status,
        "detail": detail,
        "fix": fix,
    }
    if fix_by_platform is not None:
        rec["fix_by_platform"] = fix_by_platform
    return rec


# ---------------------------------------------------------------- hard checks


def check_python() -> dict:
    name = f"python >= {'.'.join(str(p) for p in MIN_PYTHON)}"
    if sys.version_info[:2] >= MIN_PYTHON:
        return _rec("python", name, "hard", "ok", f"python {python_version()}")
    return _rec(
        "python", name, "hard", "fail",
        f"python {python_version()} ниже требуемых {'.'.join(str(p) for p in MIN_PYTHON)}",
        fix=PYTHON_FIX.get(pkg_family(), "установите Python >= 3.10"),
        fix_by_platform=dict(PYTHON_FIX),
    )


def missing_py_modules() -> list[str]:
    """Names of PY_MODULES that fail to import in this interpreter."""
    missing = []
    for mod in PY_MODULES:
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(mod)
    return missing


def check_py_modules() -> dict:
    name = "python-модули (" + " ".join(PY_MODULES) + ")"
    missing = missing_py_modules()
    if not missing:
        return _rec("modules", name, "hard", "ok", f"все {len(PY_MODULES)} импортируются")
    pip = f"python3 -m pip install {' '.join(missing)}"
    per = {
        "apt": install_cmd("apt", [MODULE_PACKAGES[m][0] for m in missing]),
        "dnf": install_cmd("dnf", [MODULE_PACKAGES[m][1] for m in missing]),
        "pacman": install_cmd("pacman", [MODULE_PACKAGES[m][2] for m in missing]),
        "brew": pip,
    }
    return _rec(
        "modules", name, "hard", "fail",
        f"не импортируются: {', '.join(missing)}",
        fix=per.get(pkg_family(), pip),
        fix_by_platform=per,
    )


def check_tools() -> list[dict]:
    family = pkg_family()
    recs = []
    for tool in TOOLS:
        name = f"{tool} в PATH"
        found = shutil.which(tool)
        if found is not None:
            recs.append(_rec(f"bin:{tool}", name, "hard", "ok", f"найден: {found}"))
            continue
        pkg = TOOL_PACKAGES[tool]
        per = {fam: install_cmd(fam, [pkg[FAMILIES.index(fam)]]) for fam in FAMILIES}
        recs.append(_rec(
            f"bin:{tool}", name, "hard", "fail", "не найден в PATH",
            fix=per.get(family, install_cmd("unknown", [pkg[0]])),
            fix_by_platform=per,
        ))
    return recs


def read_config(path: Path):
    """Parse and validate workspace.json -> (dict, "") or (None, reason).

    Valid means: JSON object with a non-empty string data_root, an integer
    budget_bytes and a dirs mapping of strings -- the keys
    rnext.workspace.load_config() and dir_of()/budget_bytes() read.
    """
    if not path.is_file():
        return None, f"файл не найден: {path}"
    try:
        conf = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"не читается как JSON: {exc}"
    if not isinstance(conf, dict):
        return None, "JSON должен быть объектом"
    problems = []
    if not isinstance(conf.get("data_root"), str) or not conf["data_root"].strip():
        problems.append("нет непустого строкового data_root")
    budget = conf.get("budget_bytes")
    if isinstance(budget, bool) or not isinstance(budget, int):
        problems.append("нет целочисленного budget_bytes")
    dirs = conf.get("dirs")
    if not isinstance(dirs, dict) or not all(isinstance(v, str) for v in dirs.values()):
        problems.append("dirs должен быть словарём строк")
    if problems:
        return None, "; ".join(problems)
    return conf, ""


def check_config(path: Path, err: str) -> dict:
    name = "config/workspace.json существует и валиден"
    if err == "":
        return _rec("config", name, "hard", "ok", str(path))
    return _rec(
        "config", name, "hard", "fail", err,
        fix=CONFIG_FIX, fix_by_platform={fam: CONFIG_FIX for fam in FAMILIES},
    )


def effective_data_root(conf: dict | None) -> Path | None:
    """Effective data_root; RESTORE_DATA_ROOT wins, as in rnext.workspace.load_config."""
    if not conf:
        return None
    raw = os.environ.get("RESTORE_DATA_ROOT") or conf.get("data_root")
    if isinstance(raw, str) and raw.strip():
        return Path(raw)
    return None


def check_data_root(root: Path | None) -> dict:
    name = "data_root существует и доступен для записи"
    if root is None:
        return _rec(
            "data_root", name, "hard", "fail",
            "data_root не определён (конфиг невалиден)",
            fix=CONFIG_FIX, fix_by_platform={fam: CONFIG_FIX for fam in FAMILIES},
        )
    if not root.is_dir():
        return _rec(
            "data_root", name, "hard", "fail", f"каталог не существует: {root}",
            fix=DATA_ROOT_FIX, fix_by_platform={fam: DATA_ROOT_FIX for fam in FAMILIES},
        )
    try:
        with tempfile.NamedTemporaryFile(prefix=".doctor-write-probe-", dir=root):
            pass
    except OSError as exc:
        return _rec(
            "data_root", name, "hard", "fail", f"нет прав на запись: {root} ({exc})",
            fix=DATA_ROOT_FIX, fix_by_platform={fam: DATA_ROOT_FIX for fam in FAMILIES},
        )
    return _rec("data_root", name, "hard", "ok", f"{root} (запись проверена)")


def check_disk(root: Path | None) -> dict:
    name = f"свободно >= {MIN_FREE_BYTES // 1024 ** 3} GiB на data_root"
    if root is None:
        return _rec(
            "disk", name, "hard", "fail", "не измерить: data_root не определён",
            fix=CONFIG_FIX, fix_by_platform={fam: CONFIG_FIX for fam in FAMILIES},
        )
    try:
        free = shutil.disk_usage(root).free
    except OSError as exc:
        return _rec(
            "disk", name, "hard", "fail", f"свободное место неизвестно: {exc}",
            fix=DISK_FIX, fix_by_platform={fam: DISK_FIX for fam in FAMILIES},
        )
    if free < MIN_FREE_BYTES:
        return _rec(
            "disk", name, "hard", "fail",
            f"свободно {free / 1024 ** 3:.2f} GiB из требуемых {MIN_FREE_BYTES / 1024 ** 3:.2f} GiB",
            fix=DISK_FIX, fix_by_platform={fam: DISK_FIX for fam in FAMILIES},
        )
    return _rec(
        "disk", name, "hard", "ok",
        f"свободно {free / 1024 ** 3:.2f} GiB (минимум {MIN_FREE_BYTES / 1024 ** 3:.2f} GiB)",
    )


# ---------------------------------------------------------------- warn checks


def secrets_path() -> Path:
    return SECRETS_DIR / SECRETS_NAME


def check_secrets() -> dict:
    path = secrets_path()
    if path.is_file():
        return _rec("secrets", "rutracker cookies", "warn", "ok", str(path))
    return _rec(
        "secrets", "rutracker cookies", "warn", "warn",
        f"tracker-сценарий недоступен, см. docs/setup.md (нет файла {path})",
        fix=f"положить cookies в {path}, см. docs/setup.md",
    )


def port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Loopback TCP probe; doctor never talks to the network beyond this host."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_transmission() -> dict:
    addr = f"{TRANSMISSION_HOST}:{TRANSMISSION_PORT}"
    name = f"transmission RPC {addr}"
    if port_open(TRANSMISSION_HOST, TRANSMISSION_PORT):
        return _rec("transmission", name, "warn", "ok", f"RPC отвечает на {addr}")
    return _rec(
        "transmission", name, "warn", "warn",
        f"скачивание недоступно (RPC {addr} не отвечает)",
        fix=f"запустить transmission-daemon с RPC на {addr}",
    )


def check_delivery_tool(tool: str, name: str, missing_detail: str,
                        fix_map: dict) -> dict:
    found = shutil.which(tool)
    if found is not None:
        return _rec(tool, name, "warn", "ok", f"найден: {found}")
    return _rec(
        tool, name, "warn", "warn", missing_detail,
        fix=fix_map.get(pkg_family(), "установите инструмент вручную"),
        fix_by_platform=dict(fix_map),
    )


def check_gio() -> dict:
    return check_delivery_tool(
        "gio", "gio (MTP-доставка)",
        "MTP-доставка недоступна (gio не найден)", GIO_FIX,
    )


def check_adb() -> dict:
    return check_delivery_tool(
        "adb", "adb (доставка по Android/ADB)",
        "доставка по ADB недоступна (adb не найден)", ADB_FIX,
    )


# ------------------------------------------------------------------ assembly


def run_checks(config_path: Path | None = None) -> list[dict]:
    """All doctor checks, in report order."""
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    checks = [check_python(), check_py_modules()]
    checks.extend(check_tools())
    conf, err = read_config(path)
    checks.append(check_config(path, err))
    root = effective_data_root(conf)
    checks.append(check_data_root(root))
    checks.append(check_disk(root))
    checks.append(check_secrets())
    checks.append(check_transmission())
    checks.append(check_gio())
    checks.append(check_adb())
    return checks


def exit_code_for(checks: list[dict]) -> int:
    if any(c["status"] == "fail" for c in checks):
        return 1
    if any(c["status"] == "warn" for c in checks):
        return 2
    return 0


def build_doc(checks: list[dict], code: int) -> dict:
    return {
        "python": python_version(),
        "platform": f"{sys.platform}/{pkg_family()}",
        "exit_code": code,
        "checks": checks,
    }


def render_table(checks: list[dict], code: int) -> str:
    marks = {"ok": "[OK]", "fail": "[FAIL]", "warn": "[WARN]"}
    name_w = max(len(c["name"]) for c in checks)
    lines = [
        f"music-restoration doctor — python {python_version()}, пакеты: {pkg_family()}",
        "",
        f"{'СТАТУС':<7}{'ПРОВЕРКА':<{name_w}}  КАК ИСПРАВИТЬ",
    ]
    for c in checks:
        fix = c["fix"] if c["status"] != "ok" else "—"
        lines.append(f"{marks[c['status']]:<7}{c['name']:<{name_w}}  {fix}")
    lines.append("")
    lines.append(f"итог: exit {code} (0 — ок; 1 — HARD-провал; 2 — только WARN)")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI boundary: parse args, run the checks, print, return the exit code."""
    parser = argparse.ArgumentParser(
        prog="doctor",
        description="Локальная диагностика окружения music-restoration (сеть не используется).",
    )
    parser.add_argument("--json", action="store_true",
                        help="вывести JSON-отчёт вместо таблицы")
    parser.add_argument("--config", type=Path, default=None,
                        help="путь к workspace.json (по умолчанию config/workspace.json репозитория)")
    args = parser.parse_args(argv)
    checks = run_checks(args.config)
    code = exit_code_for(checks)
    if args.json:
        print(json.dumps(build_doc(checks, code), ensure_ascii=False, indent=2))
    else:
        print(render_table(checks, code), end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
