import pytest
import tempfile
import os
from backend import database

@pytest.fixture(autouse=True)
def setup_db():
    # Create temp DB
    fd, path = tempfile.mkstemp()
    os.close(fd)

    original_path = database.DATABASE_PATH
    database.DATABASE_PATH = path
    database.init_database()

    yield

    # Cleanup
    os.remove(path)
    database.DATABASE_PATH = original_path

def test_raptor_clusters_crud():
    """Test Create, Read, Delete for RAPTOR clusters table."""
    # 1. Clear existing (just in case)
    database.clear_clusters()
    
    # 2. Add clusters
    c1_id = database.add_cluster("Cluster Summary 1", level=0)
    c2_id = database.add_cluster("Cluster Summary 2", level=1)
    
    # 3. Verify IDs returned
    assert c1_id is not None
    assert c2_id is not None
    assert c1_id != c2_id
    
    # 4. Get by level
    level_0 = database.get_clusters_by_level(0)
    level_1 = database.get_clusters_by_level(1)
    level_2 = database.get_clusters_by_level(2)
    
    assert len(level_0) == 1
    assert level_0[0]['summary'] == "Cluster Summary 1"
    
    assert len(level_1) == 1
    assert level_1[0]['summary'] == "Cluster Summary 2"
    
    assert len(level_2) == 0
    
    # 5. Clear
    database.clear_clusters()
    assert len(database.get_clusters_by_level(0)) == 0
    assert len(database.get_clusters_by_level(1)) == 0

def test_add_clusters_batch():
    """Test batch insertion of RAPTOR clusters."""
    # 1. Clear existing
    database.clear_clusters()

    # 2. Add batch
    data = [("Batch Summary 1", 0), ("Batch Summary 2", 1), ("Batch Summary 3", 0)]
    database.add_clusters_batch(data)

    # 3. Verify counts
    level_0 = database.get_clusters_by_level(0)
    level_1 = database.get_clusters_by_level(1)

    assert len(level_0) == 2
    assert len(level_1) == 1

    # Verify content
    summaries_0 = {c['summary'] for c in level_0}
    assert "Batch Summary 1" in summaries_0
    assert "Batch Summary 3" in summaries_0

    summaries_1 = {c['summary'] for c in level_1}
    assert "Batch Summary 2" in summaries_1

    # 4. Clear
    database.clear_clusters()
