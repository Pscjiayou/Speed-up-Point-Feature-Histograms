import subprocess
import sys
from pathlib import Path


def main():
    print("Expected Running Time: 10s")
    script = Path(__file__).with_name("icp_pfh.py")
    cmd = [sys.executable, "-u", str(script)]

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
