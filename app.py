from __future__ import annotations

import ast
import asyncio
import base64
import io
import ipaddress
import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any,Callable
from urllib.parse import urlparse
import duckdb
import pandas as pd
import requests
from dotenv import load_dotenv
from fastapi import FastAPI,Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse,HTMLResponse,JSONResponse

load_dotenv()
logging.basicConfig(level=logging.INFO)
log=logging.getLogger("analyst")

BASE = Path(__file__).parent

MODEL = os.getenv("GOOGLE_MODEL", "gemini-2.5-pro")
VISION_MODEL = os.getenv("GOOGLE_VISION_MODEL", "gemini-2.5-flash")
EXEC_TIMEOUT = int(os.getenv("EXEC_TIMEOUT_SECONDS", 60))
TOTAL_TIMEOUT = int(os.getenv("TOTAL_TIMEOUT_SECONDS", 170))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 4))
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_MB", 50)) * 1024 * 1024
MEM_LIMIT_MB = int(os.getenv("SANDBOX_MEM_MB", 2048))

app= FastAPI(title="Data Analyst Agent")

class AnalysisError(Exception):
    def __init__(self,msg:str,trace=None,attempts=None):
        super().__init__(msg)
        self.trace,self.attempts = trace or [],attempts or []


class Trace:
    def __init__(self):
        self.events: list[dict]=[]
        self._t0=time.time()

    def add(self,stage:str,status:str,detail: Any=""):
        self.events.append({"stage":stage,"status":status,
                            "detail":str(detail)[:600],"t":round(time.time()-self._t0,2)})


def _coerce_numeric(df:pd.DataFrame) ->pd.DataFrame:
    for c in df.columns:
        if df[c].dtype == object:
            s=pd.to_numeric(df[c].astype(str).str.replace(r"[,\s$%]","",regex=True),errors="coerce")
            nn = df[c].notna().sum()
            if nn and s.notna().sum() >= 0.9*nn:
                df[c]=s
     return df

def _read_csv(content: bytes, sep:str | None) -> pd.DataFrame:
    import csv
    last=None
    for enc in ("utf-8-sign","latin-1"):
        try:
            text=content.decode(enc)
        except UnicodeDecodeError as e:
            last=e
            continue
        try:
            df=pd.read_csv(io.StringIO(text),sep=sep or ",")
            if df.shape[1] >1 or sep:
                return df
            try:
                d=csv.Sniffer().sniff(text[:2000],delimiter=",;\t|").delimiter
                df2=pd.read_csv(io.StringIO(text), sep=d)
                return df2 if df2.shape[1] >1 else df
            except csv.Error:
                return df
        except Exception as e :
            last = e
    raise ValueError(f"Could not parse CSV: {last}")


def _read_json(content: bytes, name: str) -> pd.DataFrame:
    if name.endswith((".jsonl",".ndjson")):
        return pd.read_json(io.BytesIO(content),lines=True)
    obj = json.loads(content.decode("utf-8-sig"))
    if instance(obj,dict):
        lists = [v for v in obj.values() if instance(v, list) and v and 
    isinstance(v[0],dict)]
        return pd.json_normalize(lists[0]) if lists else pd.json_normalize([obj])
    return pd.json_normalize(obj)


def _read_database(content: bytes)-> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmp:
        path=os.path.join(tmp,"upload.db")
        Path(path).write_bytes(content)
        try:
            con=sqlite3.connect(path)
            tables=[r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
            if not  tables:
                raise ValueError("database has no tables")
            counts={t: con.execute(f'SELECT COUNT(*) FROM "{t}"').fetchone()[0] for t in tables}

            best=max(counts,key=counts.get)
            df=pd.read_sql_query(f'SELECT * FROM "{best}"', con)
            con.close()
            return df 
        except sqlite3.DatabaseError:
            con=duckdb.connect(path, read_only=True)
            tables=[r[0] for r in con.execute("SHOW TABLES").fetchall()]
            if not tables:
                raise ValueError("database has no tables")
            df = con.execute(f'SELECT * FROM "{tables[0]}"').fetchdf()
            con.close()
            return df


SUPPORTED=(".csv",".tsv",".xlsx",".xls",".parquet",".json",".jsonl",".ndjson",
           ".db",".sqlite",".sqlite",".sqlite3","..duckdb",".html",
           ".htm",".pdf",".png",".jpg",".jpeg",".webp")
ARCHIVES=(".zip",".tar",".tar.gz",".tgz")


def _read_archive(name:str,content: bytes,depth:int)->pd.DataFrame:
    if depth>=1:
        raise ValueError("Nested archives are not supported")
    members: list[tuple[str,Callable[[], bytes]]]=[]
    if name.endswith(".zip"):
        z=zipfile.Zipfile(io.BytesIO(content))
        if sum(i.file_size for i in z.infolist()) >10 *MAX_UPLOAD:
            raise ValueError("Archive expands to too much data")
        members=[(i.filename,(lambda i=i: z.read(i))) for i in z.infolist()
                 if not i.is_dir()]
    else:
        t=tarfile.open(fileobj=io.ByteIO(content),mode="r:*")
        members-[(m.name, (lambda m=m:t.extractfile(m).read())) for m in
t.getmembers() if m.isfile()]
    for fname, reader in members:
        if fname.lower().endswith(SUPPORTED) and not Path(fname).name.startswith("."):
            return load_dataset(fname,reader(),depth+1)
    raise ValueError("No supported data file found inside the archive")

def load_dataset(name:str,content: bytes, depth: int = 0) -> pd.DataFrame:
    n=name.lower()
    if n.endswith((".csv",".tsv")):
        return _read_csv(content,"\t" if n.endswith(".tsv") else None)
    if n.endswith((".xlsx",".xlsm",".xls")):
        return pd.read_excel(io.BytesIO(content))
    if n.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    if n.endswith((".db",".sqlite",".sqlite3",".duckdb")):
        return _read_database(content)
    if n.endswith(ARCHIVES):
        return _read_archive(n,content,depth)
    if n.endswith((".html",".htm")):
        return _largest_table(content.decode("utf-8","ignore"))
    if n.endswith(".pdf"):
        return extract_table_with_llm(content,"application/pdf")
    if n.endswith((".png",".jpg",".jpeg",".webp")):
        ext=n.rsplit(".",1)[1].replace("jpg","jpeg")
        return extract_table_with_llm(content,f"image/{ext}")
    raise ValueError(f"Unsupported file type: {name}")


def largest_table(html:str) ->pd.DataFrame:
    tables=pd.read_html(io.StringIO(html))
    if not tables:
        raise ValueError("No tables found")
    df=max(tables,key=lambda t:t.shape[0] * t.shape[1])
    df.columns=[" ".join(map(str, c)).strip() if isinstance(c,tuple) else
                str(c).strip() for c in df.columns]
    return df


#==================LLM ACCESS=====================================

_llm_cache: dict[str, Any]= {}

def get_llm(model:str):
    if model not in _llm_cache:
        key=os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("GOOGLE_API_KEY is not set")
        from langchain_google_genai import ChatGoogleGenerativeAI
        _llm_cache[model]=ChatGoogleGenerativeAI(model=model,temperature=0,google_api_key=key)
    return _llm_cache[model]


def _text(resp)->str:
    c=resp.content
    if isinstance(c, list):
        return "".join(p.get("text","") if isinstance(p, dict) else str(p) for p in c)
    return str(c or "")


def call_llm(messages: list,model: str=MODEL, retries: int =2) -> str:
    for i in range(retries+1):
        try:
            return _text(get_llm(model).invoke(messages))
        except RuntimeError:
            raise
        except Exception as e:
            if i == retries:
                raise
            log.warning("LLM call failed (%s); retry %d", e,i+1)
            time.sleep(2 ** i)

def extract_json(s:str) -> dict:
    s=re.sub(r"^```(?:json)?\s*|\s*```$", "",(s or "").strip())
    a, b=s.find("{"),s.rfind("}")
    if a == -1 or b <=a :
        raise ValueError("no JSON object found")
    return json.loads(s[a:b +1])



def extract_table_with_llm(content: bytes, mime: str) ->pd.DataFrame:
    from langchain_core.messages import HumanMessage
    media ={"type": "media", "mime_type": mime,"data":base64.b64encode(content).decode()}

    ask=('Extract ALL tabular data as JSON only: {"columns":["c1",...],"rows":[[...],[...]]}. '
         "Merge tables that share columns.Numbers must be numeric.No commentary.")

    raw= call_llm([HumanMessage(content=[{"type": "text", "text":ask}, media])],VISION_MODEL)

    try:
        obj=extract_json(raw)
        df=pd.DataFrame(obj.get("rows",[]),columns=obj.get("columns") or None )
        if df.empty:
            raise ValueError("empty table")
        return _coerce_numeric(df)
    except Exception:
        txt=call_llm([HumanMessage(content=[{"type":"text","text":"Extract all visible text. Plain text only."}, media])],VISION_MODEL)
        return pd.DataFrame({"text":[txt.strip()]})




        




                      




            

            