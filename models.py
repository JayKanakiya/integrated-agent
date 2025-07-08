from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector
from datetime import datetime

db = SQLAlchemy()

class Email(db.Model):
    __tablename__ = 'emails'
    id = db.Column(db.String, primary_key=True)
    thread_id = db.Column(db.String)
    sender = db.Column(db.String, nullable=False)          
    sender_name = db.Column(db.String, nullable=True)     
    subject = db.Column(db.String)
    date = db.Column(db.DateTime)
    snippet = db.Column(db.Text)
    body = db.Column(db.Text)
    raw = db.Column(db.JSON) 

class CalendarEvent(db.Model):
    __tablename__ = "events"
    id = db.Column(db.String, primary_key=True)
    summary = db.Column(db.String)
    start = db.Column(db.DateTime)
    end = db.Column(db.DateTime)
    raw = db.Column(db.JSON)

class Embedding(db.Model):
    __tablename__ = "embeddings"
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    doc_type = db.Column(db.String, nullable=False)
    doc_id = db.Column(db.String, nullable=False)
    vector = db.Column(Vector(1536), nullable=False)

class Task(db.Model):
    __tablename__ = 'tasks'
    id = db.Column(db.Integer, primary_key=True)
    task_type = db.Column(db.String, nullable=False)         
    parameters = db.Column(db.JSON, nullable=False)           
    status = db.Column(db.String, default='pending')          
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    related_thread = db.Column(db.String, nullable=True) 