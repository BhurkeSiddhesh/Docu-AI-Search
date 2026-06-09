import time
import os
import configparser
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from backend.indexing import create_index, save_index

# Absolute path configurations to match backend/api.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_PATH = os.path.join(BASE_DIR, 'config.ini')
INDEX_PATH = os.path.join(DATA_DIR, 'index.faiss')

class IndexingEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that rebuilds the FAISS index on file changes.

    Listens for creation, modification, and deletion events within a
    monitored folder and triggers the indexing pipeline.
    """
    def __init__(self, folder, provider, api_key, model_path, debounce_delay=0.0):
        """
        Initializes the event handler with indexing configuration.

        Args:
            folder (str or List[str]): The absolute path(s) of the directory to monitor.
            provider (str): The LLM/Embedding provider (e.g., 'local', 'openai').
            api_key (str, optional): API key for the provider.
            model_path (str, optional): Path to the local GGUF model if applicable.
            debounce_delay (float): Wait time in seconds before triggering index rebuild.
        """
        self.folder = folder
        self.provider = provider
        self.api_key = api_key
        self.model_path = model_path
        self.debounce_delay = debounce_delay
        self._lock = threading.Lock()
        self._timer = None

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        self.queue_update()

    def on_created(self, event):
        """Called when a file or directory is created."""
        self.queue_update()

    def on_deleted(self, event):
        """Called when a file or directory is deleted."""
        self.queue_update()

    def queue_update(self):
        """Queues an update, debouncing rapid successive events."""
        if not self.debounce_delay:
            self.update_index()
            return

        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_delay, self._run_update)
            self._timer.start()

    def _run_update(self):
        with self._lock:
            self._timer = None
        self.update_index()

    def update_index(self):
        """
        Triggers the indexing and saving process.

        Executes the `create_index` pipeline and persists the results to
        the absolute INDEX_PATH.
        """
        print("Change detected, updating index...")
        res = create_index(self.folder, self.provider, self.api_key, self.model_path)
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        if index:
            save_index(index, docs, tags, INDEX_PATH, idx_sum, clus_sum, clus_map, bm25)
            print("Index updated.")

def start_background_indexing():
    """
    Main background process loop that monitors the configured folders.

    Reads configuration from 'config.ini', sets up a Watchdog Observer,
    and runs a continuous loop until a KeyboardInterrupt is received.
    """
    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    if config.has_section('General') and config.getboolean('General', 'auto_index', fallback=False):
        # Handle both old 'folder' and new 'folders' format
        folders_str = config.get('General', 'folders', fallback='')
        folders = [f.strip() for f in folders_str.split(',') if f.strip()]
        if not folders:
            folder = config.get('General', 'folder', fallback='')
            if folder:
                folders = [folder]

        if not folders:
            print("No folders configured for indexing.")
            return

        provider = config.get('LocalLLM', 'provider', fallback='openai')
        api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
        model_path = config.get('LocalLLM', 'model_path', fallback=None)

        event_handler = IndexingEventHandler(folders, provider, api_key, model_path, debounce_delay=2.0)
        event_handler.update_index()
        observer = Observer()
        
        for folder in folders:
            if os.path.exists(folder):
                observer.schedule(event_handler, folder, recursive=True)
                print(f"Monitoring folder: {folder}")
            else:
                print(f"Warning: Monitored folder does not exist: {folder}")

        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    start_background_indexing()

