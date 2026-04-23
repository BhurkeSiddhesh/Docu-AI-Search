import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from backend.indexing import create_index, save_index
import configparser

class IndexingEventHandler(FileSystemEventHandler):
    """
    Watchdog event handler that rebuilds the FAISS index on file changes.

    Listens for creation, modification, and deletion events within a
    monitored folder and triggers the indexing pipeline.
    """
    def __init__(self, folder, provider, api_key, model_path):
        """
        Initializes the event handler with indexing configuration.

        Args:
            folder (str): The absolute path of the directory to monitor.
            provider (str): The LLM/Embedding provider (e.g., 'local', 'openai').
            api_key (str, optional): API key for the provider.
            model_path (str, optional): Path to the local GGUF model if applicable.
        """
        self.folder = folder
        self.provider = provider
        self.api_key = api_key
        self.model_path = model_path

    def on_modified(self, event):
        """Called when a file or directory is modified."""
        self.update_index()

    def on_created(self, event):
        """Called when a file or directory is created."""
        self.update_index()

    def on_deleted(self, event):
        """Called when a file or directory is deleted."""
        self.update_index()

    def update_index(self):
        """
        Triggers the indexing and saving process.

        Executes the `create_index` pipeline and persists the results to
        'index.faiss' and related files.
        """
        print("Change detected, updating index...")
        res = create_index(self.folder, self.provider, self.api_key, self.model_path)
        index, docs, tags, idx_sum, clus_sum, clus_map, bm25 = res[:7]
        if index:
            save_index(index, docs, tags, 'index.faiss', idx_sum, clus_sum, clus_map, bm25)
            print("Index updated.")

def start_background_indexing():
    """
    Main background process loop that monitors the configured folders.

    Reads configuration from 'config.ini', sets up a Watchdog Observer,
    and runs a continuous loop until a KeyboardInterrupt is received.
    """
    config = configparser.ConfigParser()
    config.read('config.ini')

    if config.has_section('General') and config.getboolean('General', 'auto_index', fallback=False):
        folder = config.get('General', 'folder')
        provider = config.get('LocalLLM', 'provider', fallback='openai')
        api_key = config.get('APIKeys', 'openai_api_key', fallback=None)
        model_path = config.get('LocalLLM', 'model_path', fallback=None)

        event_handler = IndexingEventHandler(folder, provider, api_key, model_path)
        event_handler.update_index()
        observer = Observer()
        observer.schedule(event_handler, folder, recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    start_background_indexing()
