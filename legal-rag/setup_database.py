import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

database_url = os.getenv("DATABASE_URL")

if not database_url:
    print("ERROR: DATABASE_URL not found in .env file")
    exit()

try:
    conn = psycopg2.connect(database_url)
    cur = conn.cursor()

    # Enable pgvector extension
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Create table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS legal_chunks (
            id bigserial PRIMARY KEY,
            text text NOT NULL,
            embedding vector(384),
            source text NOT NULL,
            collection text NOT NULL,
            chunk_hash text UNIQUE NOT NULL,
            metadata jsonb DEFAULT '{}',
            created_at timestamptz DEFAULT now()
        )
    """)

    # Create index
    cur.execute("""
        CREATE INDEX IF NOT EXISTS legal_chunks_collection_idx
        ON legal_chunks (collection)
    """)

    conn.commit()

    print("Done! Database setup completed successfully.")

    cur.close()
    conn.close()

except Exception as e:
    print("Database Error:", e)