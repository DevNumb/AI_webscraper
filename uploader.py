# src/uploader.py
import os
from supabase import create_client, Client
import pandas as pd

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
TABLE_NAME = "web_search_results"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def upload_dataframe(df: pd.DataFrame):
    # ensure JSON serializable and small batches
    records = df.to_dict(orient="records")
    # optional: chunk inserts if length large
    if not records:
        return None
    resp = supabase.table(TABLE_NAME).insert(records).execute()
    return resp
