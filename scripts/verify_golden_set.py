
import os
import requests
import time
import sys
import json
from pprint import pprint

# Backend URL
API_URL = "http://localhost:8000/api"
GOLDEN_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "golden_dataset")

def wait_for_backend():
    """
    Polls the backend API until it is ready to receive requests.
    Attempts to connect up to 10 times with a 1-second delay between tries.

    Returns:
        bool: True if the backend responds successfully, False otherwise.
    """

def configure_folder():
    """
    Submits the golden dataset path to the application's configuration.
    Ensures that the directory containing verification files is part of the 
    monitored folder list.
    """
    print(f"Adding {GOLDEN_DIR} to config...")
    # Get current config
    r = requests.get(f"{API_URL}/config")
    current_config = r.json()
    folders = current_config.get('folders', [])
    
    if GOLDEN_DIR not in folders:
        folders.append(GOLDEN_DIR)
        requests.post(f"{API_URL}/config", json={**current_config, "folders": folders})
        print("Folder added.")
    else:
        print("Folder already configured.")

def trigger_indexing():
    """
    Initiates a background indexing process and monitors its progress.
    Blocks execution until the indexing status reports that it is no longer running.
    """
    print("Triggering index...")
    requests.post(f"{API_URL}/index")
    
    # Poll status
    while True:
        r = requests.get(f"{API_URL}/index/status")
        status = r.json()
        if not status['running']:
            print("Indexing complete.")
            break
        print(f"Indexing: {status['progress']}%...")
        time.sleep(2)

def verify_query(query: str, expected_text: str, filename: str) -> bool:
    """
    Performs a search query and validates that the expected answer is retrieved.

    Checks both the raw search results (summaries/documents) and the 
    AI-generated synthesis answer for the occurrence of the 'needle' text.

    Args:
        query (str): The search question to ask.
        expected_text (str): The specific text/fact expected in the results.
        filename (str): The name of the file that should contain the information.

    Returns:
        bool: True if the expected text is found in either results or AI answer.
    """
    print(f"\nQuerying: '{query}'...")
    r = requests.post(f"{API_URL}/search", json={"query": query})
    if r.status_code != 200:
        print(f"Search failed: {r.status_code}")
        return False
        
    data = r.json()
    results = data.get('results', [])
    ai_answer = data.get('ai_answer', '')
    
    found_in_results = False
    for res in results:
        if filename in str(res.get('file_name', '')) or filename in str(res.get('file_path', '')):
            if expected_text.lower() in res.get('summary', '').lower() or expected_text.lower() in res.get('document', '').lower():
                found_in_results = True
                break
                
    found_in_answer = expected_text.lower() in ai_answer.lower()
    
    print(f"  > Found in Results: {found_in_results}")
    print(f"  > Found in AI Answer: {found_in_answer}")
    
    if found_in_results or found_in_answer:
        print("  [PASS] Needle found!")
        return True
    else:
        print(f"  [FAIL] Expected '{expected_text}' not found.")
        return False

def main():
    if not wait_for_backend():
        print("Backend not running. Please start it with 'npm run start'.")
        sys.exit(1)
        
    configure_folder()
    trigger_indexing()
    
    # Test Cases (Needle in Haystack)
    tests = [
        ("What is the population of Clouded Leopards?", "10,000", "synthetic_nature.pdf"),
        ("What is the Q3 growth in Asia?", "18%", "synthetic_sales.xlsx"),
        ("What is the Q3 goal for Project Phoenix?", "Launch MVP", "synthetic_roadmap.pptx"),
        ("What percent of enterprises plan to adopt AI agents?", "85%", "synthetic_report.docx")
    ]
    
    passed = 0
    for q, expected, fname in tests:
        if verify_query(q, expected, fname):
            passed += 1
            
    print(f"\nTotal Result: {passed}/{len(tests)} tests passed.")
    if passed == len(tests):
        print("ALL GOLDEN DATASET TESTS PASSED. 🚀")
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
