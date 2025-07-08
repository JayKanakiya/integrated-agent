import os
import json
import base64
import threading
import time
import re
import requests

from flask import (
    Flask, session, redirect, url_for,
    request, render_template, current_app, Response
)
from authlib.integrations.flask_client import OAuth
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from email.mime.text import MIMEText
from dotenv import load_dotenv
from hubspot_utils import ingest_hubspot_contacts
import openai

from datetime import datetime, timedelta
from dateutil.parser import parse as date_parse
import dateparser

from models import db, Email, CalendarEvent, Embedding, Task
from ingestion import ingest_gmail, ingest_calendar
from vectorstore import get_top_k_docs

load_dotenv(os.getenv("ENV_PATH", ".env"))
CLIENT_ID     = os.getenv("HUBSPOT_CLIENT_ID_HUBSPOT")
CLIENT_SECRET = os.getenv("HUBSPOT_CLIENT_SECRET_HUBSPOT")

openai.api_key = os.getenv("OPENAI_API_KEY")

app = Flask(__name__, template_folder="templates")
app.config.update(
    SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret"),
    SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    POLL_INTERVAL_SECONDS=int(os.getenv("POLL_INTERVAL_SECONDS", 60)),
)

db.init_app(app)
oauth = OAuth(app)
oauth.register(
    name="google",
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    client_kwargs={
        "access_type": "offline",
        "prompt": "consent",
        "scope": (
            "openid email profile "
            "https://www.googleapis.com/auth/gmail.readonly "
            "https://www.googleapis.com/auth/gmail.send "
            "https://www.googleapis.com/auth/calendar.events "
            "https://www.googleapis.com/auth/calendar.readonly "
        )
    }
)

with app.app_context():
    db.create_all()

RULES_TEXT = (
    "You are an AI assistant with access to the user's Gmail and Calendar. "
    "Answer general queries in natural language using the provided context. "
    "When asked to send an email or create a calendar event, respond only with the JSON for the corresponding function call. "
    "Do NOT ask any follow-up questions. "
    "If scheduling and no date/time is provided, include only 'summary'; server will handle availability and follow-up."
)

def _get_creds_from_config():
    tok = app.config.get("GOOGLE_TOKEN")
    if not tok:
        raise RuntimeError("Google token not found in app.config")
    return Credentials(
        token=tok["access_token"],
        refresh_token=tok.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    )

def _poll_for_slot_tasks():
    with app.app_context():
        interval = app.config["POLL_INTERVAL_SECONDS"]
        while True:
            token = app.config.get("GOOGLE_TOKEN")
            if token:
                creds = _get_creds_from_config()
                svc = build("gmail", "v1", credentials=creds)
                now = datetime.now()

                tasks = Task.query.filter_by(status="waiting_for_slot").all()
                for task in tasks:
                    params   = task.parameters
                    thread_id = params.get("thread_id")
                    if not thread_id:
                        continue

                    thread = svc.users().threads().get(
                        userId="me", id=thread_id, format="full"
                    ).execute()
                    messages = thread.get("messages", [])
                    if len(messages) <= 1:
                        current_app.logger.info(f"[Polling] No reply yet for thread {thread_id}")
                        continue

                    raw_snip = messages[-1].get("snippet", "")
                    cleaned  = raw_snip.split("On ")[0].split("\n")[0].strip()
                    current_app.logger.info(f"[Polling] Cleaned reply for {thread_id}: {cleaned}")

                    slot_dt = dateparser.parse(cleaned, settings={"PREFER_DATES_FROM":"future"})
                    if not slot_dt:
                        try:
                            slot_dt = date_parse(cleaned, default=now, fuzzy=True)
                        except Exception as e:
                            current_app.logger.warning(f"[Polling] Couldn’t parse slot: {e}")
                            continue

                    if slot_dt < now:
                        slot_dt = slot_dt.replace(year=now.year + 1)
                    start_iso = slot_dt.isoformat()
                    end_iso   = (slot_dt + timedelta(hours=1)).isoformat()

                    orig = params.get("original_args", {})
                    summary   = orig.get("summary")
                    attendees = orig.get("attendees", [])
                    if not summary:
                        current_app.logger.error(
                            f"[Polling] Missing summary in task.parameters: {params}"
                        )
                        continue

                    current_app.logger.info(
                        f"[Polling] Scheduling '{summary}' at {start_iso}"
                    )
                    ev = _create_event_internal({
                        "summary":   summary,
                        "attendees": attendees,
                        "start":     start_iso,
                        "end":       end_iso,
                        "creator_email": params.get("creator_email")
                    })
                    current_app.logger.info(
                        f"[Polling] Event created: id={ev.get('id')}"
                    )

                    task.status = "completed"
                    db.session.commit()

            time.sleep(interval)

def start_polling_thread():
    threading.Thread(target=_poll_for_slot_tasks, daemon=True).start()

@app.route("/")
def index():
    return redirect(url_for("chat_ui"))

@app.route("/chat_ui")
def chat_ui():
    if not session.get("user"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login")
def login():
    return oauth.google.authorize_redirect(url_for("auth_callback", _external=True))

@app.route("/auth/callback")
def auth_callback():
    token = oauth.google.authorize_access_token()
    session["google_token"] = token
    app.config["GOOGLE_TOKEN"] = token

    creds = _get_creds_from_config()
    ingest_gmail(creds)
    ingest_calendar(creds)

    userinfo = token.get("userinfo", {})
    session["user"] = {"email": userinfo.get("email"), "name": userinfo.get("name")}
    return redirect(url_for("chat_ui"))

@app.route("/chat", methods=["POST"])
def chat():
    pending = Task.query.filter_by(status="awaiting_contact").first()
    if pending:
        email_addr = request.json.get("message", "").strip()
        if "@" not in email_addr:
            return Response(
                "That doesn’t look like an email address. Please send a valid one, e.g. alice@example.com.",
                mimetype="text/plain"
            )

        orig = pending.parameters["original_args"]
        pending.task_type = "schedule_event"
        pending.parameters = {
            "summary":   orig["summary"],
            "attendees": [email_addr],
            "creator_email": session["user"]["email"]
        }
        pending.status = "waiting_for_slot"
        db.session.commit()

        # PATCHED: pass string, not list
        sent = _send_email_internal({
            "to":      email_addr,
            "subject": f"Availability for {orig['summary']}",
            "body":    (
                "Hi,\n\n"
                "Here are my available slots:\n"
                "- Monday 9am\n"
                "- Tuesday 2pm\n\n"
                "Please let me know which works for you.\n\nThanks!"
            )
        })
        pending.parameters["thread_id"] = sent["threadId"]
        db.session.commit()

        return Response(
            f"✅ Thanks! Emailed availability to {email_addr}—will schedule once they reply.",
            mimetype="text/plain"
        )

    user_msg = request.json.get("message", "")

    docs    = get_top_k_docs(user_msg)
    context = "\n\n".join(d["content"] for d in docs)

    messages = [
        {"role": "system",  "content": RULES_TEXT},
        {"role": "system",  "content": f"Context:\n{context}"},
        {"role": "user",    "content": user_msg}
    ]

    functions = [
        {
            "name": "send_email",
            "description": "Send an email via Gmail API",
            "parameters": {
                "type": "object",
                "properties": {
                    "to":      {"type": "string"},
                    "subject": {"type": "string"},
                    "body":    {"type": "string"}
                },
                "required": ["to", "subject", "body"]
            }
        },
        {
            "name": "create_event",
            "description": "Create a calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary":   {"type": "string"},
                    "start":     {"type": "string"},
                    "end":       {"type": "string"},
                    "attendees": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["summary"]
            }
        },
        {
            "name": "create_instruction",
            "description": "Save a new ongoing instruction for proactive workflows",
            "parameters": {
                "type": "object",
                "properties": {
                "trigger": {
                    "type": "string",
                    "description": "The event to watch for, e.g. 'email_from_unknown' or 'calendar_event_created'"
                },
                "action": {
                    "type": "string",
                    "description": "The action to take, e.g. 'email_attendees' or 'create_contact_with_note'"
                },
                "parameters": {
                    "type": "object",
                    "description": "Any extra key/value settings for this instruction"
                }
                },
                "required": ["trigger", "action"]
            }
            }
    ]
    user_msg = request.json["message"]
    if user_msg.lower().startswith("schedule"):
        fc = {"name": "create_event"}      
    else:
        fc = "auto"
    print(f"[Chat] Function call: {fc}")
    resp = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=messages,
        functions=functions,
        function_call=fc
    )
    msg = resp["choices"][0]["message"]
    print(msg, type(msg))

    if msg.get("function_call"):
        fn_name = msg["function_call"]["name"]
        args    = json.loads(msg["function_call"]["arguments"])

        if fn_name == "send_email":
            _send_email_internal(args)
            return Response(
                f"✅ Email sent to {args['to']}.",
                mimetype="text/plain"
            )

        if fn_name == "create_event":
            import re
            from sqlalchemy import or_
            from datetime import datetime, timedelta
            from googleapiclient.discovery import build

            current_app.logger.info("🟢 Entered create_event")

            # 1) Determine attendee(s)
            attendees = args.get("attendees") or []
            current_app.logger.debug(f"Raw attendees arg: {attendees}")
            if not attendees:
                m = re.search(r"[Ww]ith\s+(.+)$", args.get("summary", ""))
                if m:
                    attendees = [m.group(1).strip()]
                    current_app.logger.info(f"Extracted attendee from summary: {attendees}")
            if not attendees:
                current_app.logger.warning("No attendees found; prompting user")
                return Response(
                    "Who should attend this meeting? Please provide a name or email.",
                    mimetype="text/plain"
                )
            args["attendees"] = attendees

            # 2) Resolve names → emails (or ask user)
            resolved = []
            for name in attendees:
                current_app.logger.info(f"Resolving attendee '{name}'")
                if "@" in name:
                    current_app.logger.debug(f"'{name}' looks like an email, accepting as-is")
                    resolved.append(name)
                    continue

                contact = (
                    Email.query
                         .filter(or_(
                             Email.sender_name.ilike(f"%{name}%"),
                             Email.sender.ilike(f"%{name}%")
                         ))
                         .order_by(Email.date.desc())
                         .first()
                )
                if contact:
                    current_app.logger.info(f"Resolved '{name}' → {contact.sender}")
                    resolved.append(contact.sender)
                else:
                    current_app.logger.error(f"Could not resolve '{name}' to an email")
                    task = Task(
                        task_type="schedule_event",
                        parameters={"original_args": args},
                        status="awaiting_contact"
                    )
                    db.session.add(task)
                    db.session.commit()
                    current_app.logger.info(f"Enqueued awaiting_contact task id={task.id}")
                    return Response(
                        f"I don’t have an email for '{name}'. Could you please provide it?",
                        mimetype="text/plain"
                    )
            current_app.logger.debug(f"Final resolved attendees: {resolved}")
            args["attendees"] = resolved

            # 3) If start provided → schedule immediately
            if args.get("start"):
                current_app.logger.info(f"Start time provided: {args['start']}, scheduling now")
                _create_event_internal(args)
                current_app.logger.info("Event scheduled immediately")
                return Response(
                    f"✅ Event '{args['summary']}' scheduled on {args['start']}.",
                    mimetype="text/plain"
                )

            # 4) Otherwise, fetch real free/busy & email slots
            current_app.logger.info("No start time—fetching free/busy for next 3 days")
            creds = _get_creds_from_config()
            cal_service = build("calendar", "v3", credentials=creds)
            now = datetime.utcnow()

            fbq = {
                "timeMin": now.isoformat() + "Z",
                "timeMax": (now + timedelta(days=3)).isoformat() + "Z",
                "items": [{"id": "primary"}]
            }
            current_app.logger.debug(f"Freebusy query body: {fbq}")
            fb_res = cal_service.freebusy().query(body=fbq).execute()
            busy = fb_res["calendars"]["primary"]["busy"]
            current_app.logger.debug(f"Busy intervals: {busy}")

            def is_free(s, e):
                for iv in busy:
                    bs = datetime.fromisoformat(iv["start"].replace("Z", ""))
                    be = datetime.fromisoformat(iv["end"].replace("Z", ""))
                    if s < be and e > bs:
                        return False
                return True

            slots = []
            for d in range(3):
                day = now + timedelta(days=d)
                for h in (9, 11, 14, 16):
                    start_dt = day.replace(hour=h, minute=0, second=0, microsecond=0)
                    end_dt = start_dt + timedelta(hours=1)
                    if is_free(start_dt, end_dt):
                        slots.append(start_dt)
            current_app.logger.info(f"Computed available slots: {slots}")

            # format and send email
            lines = ["Hi,\nHere are my available slots for the next 3 days:"]
            lines += [f"- {dt.strftime('%A, %B %d at %I:%M %p')}" for dt in slots]
            lines.append("\nPlease let me know which works for you.\nThanks!")
            email_body = "\n".join(lines)
            current_app.logger.debug(f"Email body:\n{email_body}")

            to_addr = resolved[0] if len(resolved) == 1 else ", ".join(resolved)
            sent = _send_email_internal({
                "to":      to_addr,
                "subject": f"Availability for {args['summary']}",
                "body":    email_body
            })
            current_app.logger.info(f"Sent availability email, threadId={sent.get('threadId')}")

            task = Task(
                task_type="schedule_event",
                parameters={
                    "original_args": args,
                    "thread_id":     sent["threadId"],
                    "creator_email": session["user"]["email"]
                },
                status="waiting_for_slot"
            )
            db.session.add(task)
            db.session.commit()
            current_app.logger.info(f"Enqueued waiting_for_slot task id={task.id}")

            return Response(
                f"✅ Emailed availability to {to_addr}—will schedule once they reply.",
                mimetype="text/plain"
            )
    # plain-text fallback
    return Response(msg.get("content", ""), mimetype="text/plain")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("chat_ui"))

def _send_email_internal(args):
    current_app.logger.info(f"Sending email with args: {args}")
    to_field = args["to"]
    from sqlalchemy import or_

    if "@" not in to_field:
        contact = (
            Email.query
                 .filter(or_(
                     Email.sender_name.ilike(f"%{to_field}%"),
                     Email.sender.ilike(f"%{to_field}%")
                 ))
                 .order_by(Email.date.desc())
                 .first()
        )
        if contact:
            to_field = contact.sender
        else:
            raise ValueError(f"Unknown contact: {args['to']}")

    creds   = _get_creds_from_config()
    service = build("gmail", "v1", credentials=creds)
    mime    = MIMEText(args["body"])
    mime["to"]      = to_field
    mime["subject"] = args["subject"]
    raw_msg = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    return service.users().messages().send(userId="me", body={"raw": raw_msg}).execute()

def _create_event_internal(args):
    current_app.logger.info(f"Creating event with args: {args}")
    user_tz = "America/Los_Angeles"
    now = datetime.now()
    from sqlalchemy import or_

    attendees = []
    for a in args.get("attendees", []):
        if "@" not in a:
            contact = (
                Email.query
                     .filter(or_(
                         Email.sender_name.ilike(f"%{a}%"),
                         Email.sender.ilike(f"%{a}%")
                     ))
                     .order_by(Email.date.desc())
                     .first()
            )
            if contact:
                attendees.append(contact.sender)
            else:
                raise ValueError(f"Unknown contact: {a}")
        else:
            attendees.append(a)
    creator = args.get("creator_email")
    if creator:
        attendees.insert(0, creator)
    else:
        attendees.insert(0, session["user"]["email"])

    start_dt = date_parse(args["start"], default=now)
    end_dt   = date_parse(args["end"],   default=now)
    if start_dt < now:
        start_dt = start_dt.replace(year=now.year+1)
    if end_dt < now:
        end_dt = end_dt.replace(year=now.year+1)

    start_iso = start_dt.isoformat()
    end_iso   = end_dt.isoformat()

    creds   = _get_creds_from_config()
    service = build("calendar", "v3", credentials=creds)
    event_body = {
        "summary":   args["summary"],
        "start":     {"dateTime": start_iso, "timeZone": user_tz},
        "end":       {"dateTime": end_iso,   "timeZone": user_tz},
        "attendees": [{"email": e} for e in attendees]
    }
    return service.events().insert(calendarId="primary", body=event_body).execute()

if __name__ == "__main__":
    start_polling_thread()
    app.run(port=int(os.getenv("PORT", 8000)), debug=True)
