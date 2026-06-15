import subprocess
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Commande de démarage = python ".github\workflows\autowatcher.py"    

FICHIERS_SURVEILLES = [
    "FastAPI_UI.py",
]

class SaveHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        nom = event.src_path.replace('\\', '/').split('/')[-1]
        if nom not in FICHIERS_SURVEILLES:
            return
        print(f"Changement détecté : {nom}")
        subprocess.run(["git", "add", "-A"])
        result = subprocess.run(
            ["git", "commit", "-m", f"auto: {nom}"],
            capture_output=True, text=True
        )
        if "nothing to commit" not in result.stdout:
            subprocess.run(["git", "push"])
            print(f"Push effectué : {nom}")

if __name__ == "__main__":
    print("Surveillance active... (Ctrl+C pour arrêter)")
    observer = Observer()
    observer.schedule(SaveHandler(), path=".", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
