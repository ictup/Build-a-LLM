import csv, os

class CSVLogger:
    def __init__(self, path: str, fieldnames):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.DictWriter(self.f, fieldnames=fieldnames)
        self.w.writeheader()
    def log(self, **kwargs):
        self.w.writerow(kwargs); self.f.flush()
    def close(self):
        try: self.f.close()
        except: pass
