"""Download the official LJSpeech data archive (no models or executable code)."""
import hashlib
from pathlib import Path
import tarfile
import urllib.request

URL = 'https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2'
# Digest measured on the dataset used for this run, not a publisher signature.
EXPECTED_SHA256 = 'be1a30453f28eb8dd26af4101ae40cbf2c50413b1bb21936cbcdc6fae3de8aa5'


def download(root=Path('data/public')):
    archive = root / 'downloads/LJSpeech-1.1.tar.bz2'
    archive.parent.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        temporary = archive.with_suffix('.part')
        with urllib.request.urlopen(URL, timeout=60) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(archive)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != EXPECTED_SHA256:
        raise ValueError('Archive differs from the dataset used for training; inspect before extraction')
    with tarfile.open(archive, 'r:bz2') as source:
        for member in source:
            path = Path(member.name)
            metadata = member.name == 'LJSpeech-1.1/metadata.csv'
            recording = (len(path.parts) == 3 and path.parts[:2] == ('LJSpeech-1.1', 'wavs')
                         and path.suffix == '.wav')
            if member.isfile() and (metadata or recording):
                source.extract(member, root, filter='data')
    print(f'Verified dataset extracted to {root / "LJSpeech-1.1"}', flush=True)


if __name__ == '__main__':
    download()
