import hashlib
import urllib.request
import zipfile

from capstone_ids.utils import get_project_root


def download_and_unzip():
    zip_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.zip"
    md5_url = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs/MachineLearningCSV.md5"

    folder_path = get_project_root() / "data"
    zip_path = folder_path / "MachineLearningCSV.zip"
    md5_path = folder_path / "MachineLearningCSV.md5"

    folder_path.mkdir(parents=True, exist_ok=True)

    urllib.request.urlretrieve(md5_url, md5_path)

    with urllib.request.urlopen(zip_url) as response, open(zip_path, "wb") as out_file:
        while chunk := response.read(8192):
            out_file.write(chunk)

    with open(md5_path, "r") as f:
        expected_md5 = f.read().strip().split()[0]

    if not verify_md5(zip_path, expected_md5):
        raise ValueError(f"MD5 verification failed for {zip_path}")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)


def verify_md5(file_path, md5):
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest() == md5


if __name__ == "__main__":
    download_and_unzip()
