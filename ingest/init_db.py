# ingest/init_db.py
from ingest.db import DB

db = DB()
db.reset()   # fresh, empty collections
print("collections ready:", [c.name for c in db.client.get_collections().collections])
db.close()