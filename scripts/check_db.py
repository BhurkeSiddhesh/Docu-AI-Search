import sqlite3
import os

BASE_DIR = os.getcwd()
DATABASE_PATH = os.path.join(BASE_DIR, 'data', 'metadata.db')

conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(files)")
print(cursor.fetchall())
conn.close()
