import pytest
import sys
import os
import tempfile
import shutil

# Ensure backend can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend import database

@pytest.fixture(scope="session", autouse=True)
def setup_test_database():
    """
    Global database setup for all tests.
    Ensures that a temporary database is used and initialized,
    preventing tests from touching the production database.
    """
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    original_db_path = database.DATABASE_PATH

    # Set DB path to temp file
    temp_db_path = os.path.join(temp_dir, 'test_metadata.db')
    database.DATABASE_PATH = temp_db_path

    # Initialize DB (create tables)
    try:
        database.init_database()
    except Exception as e:
        print(f"Failed to initialize test database: {e}")
        # We continue, as individual tests might handle this or fail gracefully

    yield

    # Cleanup
    database.DATABASE_PATH = original_db_path
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

def pytest_sessionstart(session):
    session.results = dict()

@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    result = outcome.get_result()

    if result.when == 'call':
        item.session.results[item] = result

def pytest_collection_finish(session):
    # Store the total number of items collected
    session.total_test_count = len(session.items)
    session.tests_completed = 0
    print(f"\n[PROGRESS] Collected {session.total_test_count} tests.")

def pytest_runtest_logreport(report):
    # Only report progress after the "call" phase (actual test execution) is done
    if report.when == 'call':
        # Get the session from the plugin manager or context if possible, 
        # but report doesn't easily link back to session variables dynamically in a clean way 
        # without global state or using the item.
        pass

# Simpler approach: use pytester hook or runtest_teardown
def pytest_runtest_teardown(item, nextitem):
    session = item.session
    if not hasattr(session, 'tests_completed'):
        session.tests_completed = 0
    if not hasattr(session, 'total_test_count'):
         # Fallback if collection hook didn't run as expected or for safety
        session.total_test_count = len(session.items)
        
    session.tests_completed += 1
    
    percent = (session.tests_completed / session.total_test_count) * 100
    
    # Using sys.stdout.write to bypass capture if possible, 
    # but pytest captures stdout. using -s or -p no:capture allows seeing this.
    # We'll use a specific prefix so the user sees it.
    print(f"\n[PROGRESS] {percent:.1f}% Complete ({session.tests_completed}/{session.total_test_count}) - {item.name}")
