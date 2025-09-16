from joblib import dump
import os


def save_local(obj, path, filename):
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}.joblib")

    dump(obj, full_path)
    print(f"Salvo em {full_path}")
