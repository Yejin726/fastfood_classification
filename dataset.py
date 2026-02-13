import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
import zipfile


device = "cuda" if torch.cuda.is_available() else "cpu"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.info("Using device: %s", device)


def mount_colab_drive(mount_point: Path) -> None:
    try:
        from google.colab import drive
    except ImportError:
        logging.info("Google Colab drive mount is not available in this environment.")
        return

    logging.info("Mounting Google Drive at %s", mount_point)
    drive.mount(str(mount_point), force_remount=False)


def extract_zip_file(zip_path: Path, output_dir: Path) -> None:
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    logging.info("Extracting %s into %s", zip_path, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(output_dir)


def load_image_array(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(image_size, Image.LANCZOS)
        return np.asarray(image, dtype="float32") / 255.0


def load_labeled_images(base_dir: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    class_dirs = sorted([entry for entry in base_dir.iterdir() if entry.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories found in {base_dir}")

    classes = {idx: entry.name for idx, entry in enumerate(class_dirs)}
    name_to_idx = {name: idx for idx, name in classes.items()}

    data = []
    labels = []

    for class_dir in class_dirs:
        label = name_to_idx[class_dir.name]
        for image_file in sorted(class_dir.iterdir()):
            if not image_file.is_file():
                continue
            data.append(load_image_array(image_file, image_size))
            labels.append(label)

    return np.stack(data), np.array(labels, dtype=np.int64), classes


def load_unlabeled_images(base_dir: Path, image_size: tuple[int, int]) -> np.ndarray:
    image_files = [entry for entry in base_dir.iterdir() if entry.is_file()]
    image_files = sorted(image_files, key=lambda entry: int(entry.stem))

    dataset = [load_image_array(path, image_size) for path in image_files]
    return np.stack(dataset)


def plot_label_distribution(labels: np.ndarray, classes: dict[int, str]) -> None:
    label_names = [classes[int(label)] for label in labels.tolist()]
    df = pd.DataFrame({"label": label_names})

    plt.figure(figsize=(12, 4))
    ax = sns.countplot(x="label", data=df, order=list(classes.values()))
    ax.set_title("Training label distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def prepare_data_from_drive(drive_data_root: str, dest_root: Path, image_size: tuple[int, int] = (64, 64)) -> None:
    drive_root = Path(drive_data_root)
    if not drive_root.exists():
        raise FileNotFoundError(f"Provided Google Drive path does not exist: {drive_root}")

    train_zip = drive_root / "Train.zip"
    test_zip = drive_root / "Test.zip"

    if not train_zip.exists() or not test_zip.exists():
        raise FileNotFoundError("Train.zip and Test.zip must exist within the provided drive directory.")

    logging.info("Preparing training split")
    train_dir = dest_root / "Train"
    extract_zip_file(train_zip, train_dir)

    logging.info("Preparing test split")
    test_dir = dest_root / "Test"
    extract_zip_file(test_zip, test_dir)

    data, labels, classes = load_labeled_images(train_dir, image_size)
    test_data = load_unlabeled_images(test_dir, image_size)

    logging.info("Training data shape: %s", data.shape)
    logging.info("Training labels shape: %s", labels.shape)
    logging.info("Test data shape: %s", test_data.shape)

    unique_labels, counts = np.unique(labels, return_counts=True)
    logging.info("Number of classes: %s", unique_labels.size)
    logging.info("Samples per class: %s", counts.tolist())

    plot_label_distribution(labels, classes)


def main() -> None:
    mount_colab_drive(Path("/content/drive"))

    drive_data_path = "https://drive.google.com/drive/folders/1eW6T-zy93jW1frGAKJ8pf8_bbxc8pqdb?usp=sharing"
    target_dir = Path("./data")
    target_dir.mkdir(parents=True, exist_ok=True)

    prepare_data_from_drive(drive_data_path, target_dir)


if __name__ == "__main__":
    main()

