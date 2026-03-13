"""
Qwen 3.5 35B A3B Model Diagnostic

Verifies the integration and performance of the Qwen 3.5 35B model.
Handles automatic model download, environment readiness checks, 
and triggers a test generation with a mocked RAG context.
"""
import os
import sys
import time

# Ensure we can import from root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.llm_integration import generate_ai_answer
from backend.model_manager import MODELS_DIR, start_download, get_download_status

MODEL_ID = "Qwen3.5-35B-A3B-UD-Q4_K_M"
MODEL_PATH = os.path.join(MODELS_DIR, f"{MODEL_ID}.gguf")

def main():
    """
    Diagnostic script for testing the Qwen 3.5 35B model in a local environment.

    The script confirms the existence of the model (downloading it if necessary), 
    sets up a mockup RAG context, and triggers a text generation to verify 
    performance and resource compatibility (RAM/VRAM).
    """
    print("=== Testing Qwen 3.5 35B A3B Model ===")
    print(f"Expected path: {MODEL_PATH}")

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}.")
        print("Starting download...")
        success, msg = start_download(MODEL_ID)
        if not success:
            print(f"Failed to start download: {msg}")
            # If it failed due to system resource checks (e.g. not enough space/RAM), we bypass for testing
            # Since this is a test script, we might just print the message and exit if we can't get the file.
            return

        print("Waiting for download to complete (this will take a while for 20GB)...")
        while True:
            status = get_download_status()
            if not status["downloading"]:
                if status["error"]:
                    print(f"Download error: {status['error']}")
                    return
                break
            progress = status.get('progress', 0)
            sys.stdout.write(f"\rProgress: {progress}% ({status.get('bytes_downloaded', 0) / (1024**2):.1f} MB)")
            sys.stdout.flush()
            time.sleep(2)
        print("\nDownload finished.")

    # We create a dummy indexed context since setting up the full FAISS pipeline
    # just for a single test is overly complex for a simple LLM test

    context = """
[Document 1] Employee Handbook 2024.pdf
The Qwen team has released their latest Qwen3.5 35B A3B model, designed to be highly capable in reasoning and coding tasks.
It fits perfectly between the 14B and 72B models, offering a great balance of performance and resource usage.
The model was officially trained on multilingual data.

[Document 2] Project Roadmap.docx
Phase 1 includes testing the Qwen 35B class model for local RAG deployments.
    """

    question = "What is the new Qwen model designed to be highly capable in, and what data was it trained on?"

    print("\n--- Test Details ---")
    print(f"Model: {MODEL_ID}")
    print(f"Context:\n{context}")
    print(f"Question: {question}")
    print("--------------------")

    print("\nGenerating Answer... (This will crash if RAM is insufficient)")
    try:
        start_time = time.time()
        answer = generate_ai_answer(
            context=context,
            question=question,
            provider="local",
            model_path=MODEL_PATH,
            max_tokens=256
        )
        duration = time.time() - start_time
        print(f"\nTime taken: {duration:.2f} seconds")
        print("\n=== AI Answer ===")
        print(answer)
        print("=================")
    except Exception as e:
        print(f"\nError during generation: {e}")

if __name__ == "__main__":
    main()
