"""Serve the local transcription helper and training audio on loopback only."""
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import argparse
import re


class AudioHandler(SimpleHTTPRequestHandler):
    """Byte ranges make jumping to a sentence work without downloading from zero."""
    def send_head(self):
        path = Path(self.translate_path(self.path))
        self.remaining = None
        if path.suffix.lower() != '.wav' or not path.is_file():
            return super().send_head()
        size = path.stat().st_size
        start, end = 0, size-1
        header = self.headers.get('Range')
        if header:
            match = re.fullmatch(r'bytes=(\d*)-(\d*)', header)
            if not match or not any(match.groups()):
                self.send_error(416, 'Invalid range')
                return None
            left, right = match.groups()
            if left:
                start = int(left)
                end = min(int(right),size-1) if right else size-1
            else:
                start = max(0,size-int(right))
            if start>end or start>=size:
                self.send_response(416)
                self.send_header('Content-Range',f'bytes */{size}')
                self.send_header('Content-Length','0')
                self.end_headers()
                return None
        stream = path.open('rb')
        stream.seek(start)
        self.remaining = end-start+1
        self.send_response(206 if header else 200)
        self.send_header('Content-Type','audio/wav')
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(self.remaining))
        if header:
            self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        return stream

    def copyfile(self, source, outputfile):
        if self.remaining is None:
            return super().copyfile(source,outputfile)
        try:
            while self.remaining>0:
                chunk = source.read(min(65536,self.remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                self.remaining-=len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]/'data/training'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1',args.port), partial(AudioHandler,directory=str(root)))
    print(f'http://127.0.0.1:{server.server_port}/transcribe_video.html',flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
