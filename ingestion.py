import os
import json
import base64
import email
import time
import openai
from datetime import datetime
from http.client import RemoteDisconnected
from urllib3.exceptions import ProtocolError
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from dateutil import parser as date_parser
from models import db, Email, CalendarEvent, Embedding
from vectorstore import vectordb  
from langchain.schema import Document

MAX_EMBED_CHARS = 2000
RETRY_LIMIT = 3
RETRY_DELAY = 2  

def extract_body(payload):
    parts = payload.get('parts') or []
    for part in parts:
        if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
            data = part['body']['data']
            return base64.urlsafe_b64decode(data.encode('utf-8')).decode('utf-8')
    data = payload.get('body', {}).get('data')
    if data:
        return base64.urlsafe_b64decode(data.encode('utf-8')).decode('utf-8')
    return ''

def safe_execute(func, *args, **kwargs):
    for attempt in range(RETRY_LIMIT):
        try:
            return func(*args, **kwargs)
        except (ProtocolError, RemoteDisconnected) as e:
            print(f"Network error on attempt {attempt+1}: {e}")
            time.sleep(RETRY_DELAY)
    raise

def ingest_gmail(creds: Credentials, max_results: int = 50):
    service = build('gmail', 'v1', credentials=creds)
    try:
        results = safe_execute(
            service.users().messages().list,
            userId='me', maxResults=max_results
        ).execute()
    except Exception as e:
        print(f"Failed to list Gmail messages: {e}")
        return

    messages = results.get('messages', [])
    for m in messages:
        try:
            msg = safe_execute(
                service.users().messages().get,
                userId='me', id=m['id'], format='full'
            ).execute()
        except Exception as e:
            print(f"Skipping message {m['id']} due to fetch error: {e}")
            continue

        snippet = msg.get('snippet', '')
        payload = msg.get('payload', {})
        headers = payload.get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), None)
        from_hdr = next((h['value'] for h in headers if h['name'] == 'From'), '')
        # Parse name and email address
        name, addr = email.utils.parseaddr(from_hdr)
        date_str = next((h['value'] for h in headers if h['name'] == 'Date'), None)
        date_obj = date_parser.parse(date_str) if date_str else None
        body = extract_body(payload)

        email_rec = Email(
            id=msg['id'],
            thread_id=msg.get('threadId'),
            sender=addr,
            sender_name=name,
            subject=subject,
            date=date_obj,
            snippet=snippet,
            body=body,
            raw=msg
        )
        db.session.merge(email_rec)
        db.session.commit()

        full_text = ' '.join(filter(None, [
            f"From: {name} <{addr}>",
            subject,
            snippet,
            body
        ]))
        text_to_embed = full_text[:MAX_EMBED_CHARS]
        try:
            emb_resp = openai.Embedding.create(
                input=text_to_embed,
                model='text-embedding-ada-002'
            )
            vector = emb_resp['data'][0]['embedding']

            # Upsert Embedding model
            embedding = Embedding(
                doc_type='email', doc_id=msg['id'], vector=vector
            )
            db.session.merge(embedding)
            db.session.commit()

            # Upsert into LangChain vectorstore
            doc = Document(
                page_content=text_to_embed,
                metadata={
                    'doc_type': 'email',
                    'doc_id': msg['id'],
                    'sender_name': name,
                    'sender_email': addr
                }
            )
            vectordb.add_documents([doc])
        except Exception as e:
            print(f"Embedding skipped for email {msg['id']}: {e}")
            continue

    print(f"Ingested {len(messages)} emails.")

def ingest_calendar(creds: Credentials, max_results: int = 10):
    from datetime import datetime
    service = build('calendar', 'v3', credentials=creds)
    try:
        now_iso = datetime.utcnow().isoformat() + 'Z'
        events_result = safe_execute(
            service.events().list,
            calendarId='primary', timeMin=now_iso,
            maxResults=max_results, singleEvents=True,
            orderBy='startTime'
        ).execute()
    except Exception as e:
        print(f"Failed to list Calendar events: {e}")
        return

    events = events_result.get('items', [])
    for ev in events:
        start_raw = ev.get('start', {}).get('dateTime') or ev.get('start', {}).get('date')
        end_raw = ev.get('end', {}).get('dateTime') or ev.get('end', {}).get('date')
        start_dt = date_parser.parse(start_raw) if start_raw else None
        end_dt = date_parser.parse(end_raw) if end_raw else None
        attendees = [a.get('email') for a in ev.get('attendees', [])]

        # Upsert CalendarEvent record
        event_rec = CalendarEvent(
            id=ev['id'], summary=ev.get('summary'), start=start_dt, end=end_dt
        )
        event_rec.attendees = json.dumps(attendees)
        db.session.merge(event_rec)
        db.session.commit()

        # Prepare text for embedding
        title_desc = ' '.join(filter(None, [ev.get('summary'), ev.get('description')]))
        text_to_embed = title_desc[:MAX_EMBED_CHARS]
        try:
            emb_resp = openai.Embedding.create(
                input=text_to_embed, model='text-embedding-ada-002'
            )
            vector = emb_resp['data'][0]['embedding']
            embedding = Embedding(
                doc_type='event', doc_id=ev['id'], vector=vector
            )
            db.session.merge(embedding)
            db.session.commit()

            doc = Document(
                page_content=text_to_embed,
                metadata={'doc_type': 'event', 'doc_id': ev['id']}
            )
            vectordb.add_documents([doc])
        except Exception as e:
            print(f"Embedding skipped for event {ev['id']}: {e}")

    print(f"Ingested {len(events)} calendar events.")