"""Windows desktop launcher for the locally hosted MixSIAPy GUI."""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.request
import webbrowser


def _app_path() -> Path:
    """Return the bundled or source-tree Streamlit application path."""
    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS"))
        return bundle_root / "mixsiapy" / "gui" / "app.py"
    return Path(__file__).resolve().parent / "app.py"


def _free_port() -> int:
    """Reserve an available localhost port for the Streamlit server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(port: int) -> list[str]:
    if getattr(sys, "frozen", False):
        return [sys.executable, "--mixsiapy-server", "--port", str(port)]
    return [sys.executable, str(Path(__file__).resolve()), "--mixsiapy-server", "--port", str(port)]


def _run_server(port: int) -> None:
    """Run Streamlit in the dedicated server subprocess."""
    from mixsiapy.gui import _configure_ca_bundle

    _configure_ca_bundle()
    from streamlit.web import bootstrap

    options = {
        "global_developmentMode": False,
        "server_address": "127.0.0.1",
        "server_port": port,
        "server_headless": True,
        "browser_gatherUsageStats": False,
    }
    bootstrap.load_config_options(options)
    bootstrap.run(
        str(_app_path()),
        is_hello=False,
        args=[],
        flag_options=options,
    )


def _wait_for_server(url: str, process: subprocess.Popen, timeout: float = 90.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and process.poll() is None:
        try:
            with urllib.request.urlopen(f"{url}/_stcore/health", timeout=1.0) as response:
                if response.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def _terminate(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        process.kill()


def _run_launcher() -> int:
    """Show a small controller while the browser-based GUI is running."""
    import tkinter as tk
    from tkinter import messagebox

    port = _free_port()
    url = f"http://127.0.0.1:{port}"
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    process = subprocess.Popen(_server_command(port), creationflags=creationflags)

    root = tk.Tk()
    root.title("MixSIAPy")
    root.geometry("430x190")
    root.resizable(False, False)

    title = tk.Label(root, text="MixSIAPy local graphical interface", font=("Segoe UI", 13, "bold"))
    title.pack(pady=(22, 6))
    status = tk.StringVar(value="Starting the local analysis server …")
    tk.Label(root, textvariable=status, font=("Segoe UI", 10)).pack(pady=4)

    button_frame = tk.Frame(root)
    button_frame.pack(pady=16)
    open_button = tk.Button(
        button_frame,
        text="Open interface",
        width=16,
        state=tk.DISABLED,
        command=lambda: webbrowser.open(url),
    )
    open_button.pack(side=tk.LEFT, padx=7)

    def close() -> None:
        _terminate(process)
        root.destroy()

    tk.Button(button_frame, text="Stop and close", width=16, command=close).pack(side=tk.LEFT, padx=7)
    tk.Label(
        root,
        text="Closing this window stops the local MixSIAPy service.",
        font=("Segoe UI", 9),
        fg="#5F6B73",
    ).pack()
    root.protocol("WM_DELETE_WINDOW", close)

    def start_and_open() -> None:
        if _wait_for_server(url, process):
            status.set(f"Running locally at {url}")
            root.after(0, lambda: open_button.config(state=tk.NORMAL))
            webbrowser.open(url)
        else:
            status.set("The local service could not be started.")
            root.after(
                0,
                lambda: messagebox.showerror(
                    "MixSIAPy",
                    "The local graphical interface could not be started.\n"
                    "Please check the release notes or report the problem.",
                ),
            )

    threading.Thread(target=start_and_open, daemon=True).start()

    def monitor() -> None:
        if process.poll() is not None:
            status.set("The local analysis service has stopped.")
            open_button.config(state=tk.DISABLED)
        else:
            root.after(1000, monitor)

    root.after(1000, monitor)
    root.mainloop()
    return 0


def main() -> int:
    """Dispatch to the desktop controller or its Streamlit subprocess."""
    if "--mixsiapy-server" in sys.argv:
        index = sys.argv.index("--port")
        try:
            _run_server(int(sys.argv[index + 1]))
        except BaseException:
            log_path = Path(tempfile.gettempdir()) / "MixSIAPy-error.log"
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
            raise
        return 0
    return _run_launcher()


if __name__ == "__main__":
    raise SystemExit(main())
