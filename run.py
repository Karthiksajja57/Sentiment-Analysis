import os
import sys
import subprocess
import time
import webbrowser

def main():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(root_dir, "ML")
    frontend_dir = os.path.join(ml_dir, "frontend")

    python_exe = sys.executable

    print("=" * 55)
    print("   EnsembleIQ Sentiment Analysis Studio")
    print("=" * 55)

    # 1. Start Backend FastAPI (uvicorn) on port 8000
    print("[1/2] Starting Backend API on http://localhost:8000 ...")
    backend_proc = subprocess.Popen(
        [python_exe, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ml_dir
    )

    # 2. Start Frontend Server on port 3000
    print("[2/2] Starting Frontend Studio on http://localhost:3000 ...")
    frontend_proc = subprocess.Popen(
        [python_exe, "serve.py"],
        cwd=frontend_dir
    )

    # Allow servers a brief moment to initialize
    time.sleep(2)
    print("\n[OK] Both services are running successfully!")
    print("  * Frontend UI:   http://localhost:3000")
    print("  * Backend API:   http://localhost:8000")
    print("  * API Docs:      http://localhost:8000/docs")
    print("\nOpening frontend in your default web browser...")
    try:
        webbrowser.open("http://localhost:3000")
    except Exception:
        pass

    print("\nPress Ctrl+C at any time to stop both servers.\n")

    try:
        backend_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping servers...")
    finally:
        backend_proc.terminate()
        frontend_proc.terminate()
        backend_proc.wait()
        frontend_proc.wait()
        print("All servers stopped successfully.")

if __name__ == "__main__":
    main()
