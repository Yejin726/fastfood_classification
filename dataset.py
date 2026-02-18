import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )
    if p.stdout.strip():
        print(p.stdout.strip())


def ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(
            f"Required tool '{name}' not found. Install it first."
        )


def download_file(drive_url: str, out_dir: str = "dataset", out_name: str = "Train.zip") -> Path:
    ensure_tool("gdown")
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    out_path = out_dir_p / out_name

    # 핵심: -O 로 저장 경로를 지정
    if not out_path.exists():
        run(["gdown", "--fuzzy", drive_url, "-O", str(out_path)])

    print(f"[ok] downloaded to: {out_path.resolve()}")
    return out_path

def unzip_file(zip_path, extract_dir):
    ensure_tool("unzip")
    
    zip_path = Path(zip_path)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} does not exist.")
    
    if extract_dir is None:
        extract_dir = zip_path.parent
        
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    extracted_flag = any(extract_dir.iterdir())

    if extracted_flag:
        print(f"[skip] {extract_dir} already has contents. Skipping unzip.")
        return extract_dir
    
    print(f"[unzip] Extracting {zip_path} -> {extract_dir}")

    run(["unzip", "-o", str(zip_path), "-d", str(extract_dir)])


if __name__ == "__main__":
    DRIVE_TRAIN_URL = "https://drive.google.com/file/d/1RCl-k2s-mN0siVVyHXmE8fTC-rPzNMnD/view?usp=sharing"
    DRIVE_TEST_URL = "https://drive.google.com/file/d/1PYnRVNE_D0VEdVFsrv-RDucm27sEpt9-/view?usp=sharing"
    out_path = './data'
    
    train_unzip = download_file(DRIVE_TRAIN_URL, "./dataset", "Train.zip")
    test_unzip = download_file(DRIVE_TEST_URL, "./dataset", "Test.zip")
    
    unzip_file(train_unzip, os.path.join(out_path, 'Train'))
    unzip_file(test_unzip, os.path.join(out_path, 'Test'))
