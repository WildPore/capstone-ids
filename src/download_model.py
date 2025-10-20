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

    print("Downloading MD5 checksum...")
    urllib.request.urlretrieve(md5_url, md5_path)

    print("Downloading zip file...")
    with urllib.request.urlopen(zip_url) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0

        with open(zip_path, "wb") as out_file:
            while chunk := response.read(8192):
                out_file.write(chunk)
                downloaded += len(chunk)
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    print(
                        f"\rProgress: {downloaded / (1024**2):.1f}MB / {total_size / (1024**2):.1f}MB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
        print()

    with open(md5_path, "r") as f:
        expected_md5 = f.read().strip().split()[0]

    print("Verifying MD5 checksum...")
    if not verify_md5(zip_path, expected_md5):
        raise ValueError(f"MD5 verification failed for {zip_path}")
    print("MD5 verification passed!")

    print("Extracting files...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(folder_path)
    print("Extraction complete!")


def verify_md5(file_path, md5):
    md5_hash = hashlib.md5()

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            md5_hash.update(chunk)

    return md5_hash.hexdigest() == md5


if __name__ == "__main__":
    download_and_unzip()
