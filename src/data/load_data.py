from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

COLUMNS = (
    ["unit", "cycle"]
    + [f"setting_{i}" for i in range(1, 4)]
    + [f"sensor_{i}" for i in range(1, 22)]
)

def load_file(filename):
    path = RAW_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=COLUMNS
    )

    return df

def load_data():
    train = load_file("train_FD004.txt")
    test = load_file("test_FD004.txt")

    rul_path = RAW_DIR / "RUL_FD004.txt"

    if not rul_path.exists():
        raise FileNotFoundError(f"Missing RUL file: {rul_path}")

    test_rul = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["RUL"]
    )

    return train, test, test_rul


if __name__ == "__main__":
    train, test, test_rul = load_data()

    print("Training shape:", train.shape)
    print("Testing shape:", test.shape)
    print("Test RUL shape:", test_rul.shape)

    print("\nTraining sample:")
    print(train.head())

    print("\nTest sample:")
    print(test.head())