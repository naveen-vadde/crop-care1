"""
MongoDB Atlas data layer for CropCare (PyMongo).

Collections
-----------
users            farmer accounts
plants           plants a farmer is tracking
disease_records  one disease-detection result per plant per observation date

All functions here return plain dicts (or the `User` object for Flask-Login),
so the rest of the app never has to deal with MongoDB-specific types.
"""
import logging
import time
import uuid
from datetime import datetime, timezone

from bson import ObjectId
from bson.errors import InvalidId
from flask_login import UserMixin
from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.errors import (
    ConfigurationError,
    DuplicateKeyError,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)
from werkzeug.security import check_password_hash, generate_password_hash

import config

log = logging.getLogger(__name__)

__all__ = [
    'DuplicateKeyError', 'PyMongoError', 'User', 'init_db', 'ping',
    'create_user', 'get_user_by_id', 'get_user_by_mobile',
    'create_plant', 'get_plant', 'list_plants', 'delete_plant',
    'add_record', 'list_records', 'get_record', 'delete_record',
    'record_to_dict', 'plant_to_dict',
]

_client = None
_db = None


def utcnow():
    """Naive UTC timestamp (what PyMongo reads/writes by default)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_object_id(value):
    """Convert a string to ObjectId, or None if it is not a valid id."""
    try:
        return ObjectId(str(value))
    except (InvalidId, TypeError):
        return None


# --------------------------------------------------------------- connection
def explain_connection_error(exc):
    """Turn a PyMongo error into a message a beginner can act on."""
    text = str(exc)
    if 'DNS' in text or 'resolution' in text.lower() or 'nodename' in text.lower():
        return ("Could not resolve the Atlas cluster address from MONGODB_URI. Check the cluster "
                "host name in the connection string, your internet/DNS connection, and that the "
                "'dnspython' package is installed (pip install -r requirements.txt).")
    if isinstance(exc, ConfigurationError) or 'Invalid URI' in text or 'URI must' in text:
        return ("MONGODB_URI is not a valid MongoDB connection string. Copy it again from "
                "Atlas (Connect -> Drivers) and make sure special characters in the password "
                "are URL-encoded (e.g. @ becomes %40).")
    if isinstance(exc, OperationFailure) and (exc.code == 18 or 'auth' in text.lower()):
        return ("Atlas rejected the username/password in MONGODB_URI. Check the Database "
                "Access user in Atlas and re-enter the password (URL-encode special characters).")
    if isinstance(exc, ServerSelectionTimeoutError):
        return ("Could not reach MongoDB Atlas. The most common cause is that your current IP "
                "address is not allowed: in Atlas open Security -> Network Access -> Add IP "
                "Address (use 'Add Current IP Address', or 0.0.0.0/0 for testing). "
                "Also check your internet connection and that the cluster is not paused.")
    return f"MongoDB connection failed: {text.splitlines()[0] if text else exc!r}"


def _make_client():
    if not config.MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is not set. Add your MongoDB Atlas connection string to the .env "
            "file (see .env.example and the README)."
        )
    kwargs = dict(
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        appname='cropcare',
    )
    uri = config.MONGODB_URI
    # Atlas always uses TLS. Use certifi's CA bundle so TLS works on machines
    # whose system certificates are old/missing (common on Windows/macOS Python).
    if uri.startswith('mongodb+srv://') or 'tls=true' in uri.lower() or 'ssl=true' in uri.lower():
        try:
            import certifi
            kwargs['tlsCAFile'] = certifi.where()
        except ImportError:
            pass
    return MongoClient(uri, **kwargs)


def ping():
    """Raise if the database cannot be reached."""
    get_db().command('ping')
    return True


def get_db():
    if _db is None:
        raise RuntimeError("Database not initialised - call init_db(app) first.")
    return _db


def create_indexes():
    """Idempotent: safe to run on every start."""
    d = get_db()
    d.users.create_index([('mobile', ASCENDING)], unique=True, name='uniq_mobile')
    d.plants.create_index([('plant_id', ASCENDING)], unique=True, name='uniq_plant_code')
    d.plants.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)], name='plants_by_user')
    d.disease_records.create_index([('plant_ref', ASCENDING), ('timestamp', DESCENDING)],
                                   name='records_by_plant_time')
    d.disease_records.create_index([('user_id', ASCENDING)], name='records_by_user')


def init_db(app=None, retries=3, delay=2):
    """Connect to Atlas, verify the connection and create indexes."""
    global _client, _db
    last = None
    for attempt in range(1, retries + 1):
        try:
            if _client is None:
                _client = _make_client()
            _db = _client[config.MONGODB_DB]
            _db.command('ping')
            create_indexes()
            log.info("Connected to MongoDB (database '%s')", config.MONGODB_DB)
            return _db
        except RuntimeError:
            raise
        except (ConfigurationError, OperationFailure) as exc:
            # Wrong URI or wrong credentials: retrying will not help
            raise RuntimeError(explain_connection_error(exc)) from exc
        except PyMongoError as exc:
            last = exc
            log.warning("MongoDB connection attempt %s/%s failed: %s",
                        attempt, retries, str(exc).splitlines()[0])
            time.sleep(delay)
    raise RuntimeError(explain_connection_error(last)) from last


# -------------------------------------------------------------------- users
class User(UserMixin):
    """Flask-Login user backed by a MongoDB document."""

    def __init__(self, doc):
        self.id = str(doc['_id'])
        self.oid = doc['_id']
        self.name = doc['name']
        self.mobile = doc['mobile']
        self.password_hash = doc['password_hash']
        self.created_at = doc.get('created_at')

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


def create_user(name, mobile, password):
    """Insert a user. Raises DuplicateKeyError if the mobile is already registered."""
    doc = {
        'name': name,
        'mobile': mobile,
        'password_hash': generate_password_hash(password),
        'created_at': utcnow(),
    }
    doc['_id'] = get_db().users.insert_one(doc).inserted_id
    return User(doc)


def get_user_by_id(user_id):
    oid = to_object_id(user_id)
    if oid is None:
        return None
    doc = get_db().users.find_one({'_id': oid})
    return User(doc) if doc else None


def get_user_by_mobile(mobile):
    doc = get_db().users.find_one({'mobile': mobile})
    return User(doc) if doc else None


# ------------------------------------------------------------------- plants
def plant_to_dict(doc, total_records=0):
    return {
        'id': str(doc['_id']),
        'plant_id': doc['plant_id'],
        'plant_name': doc['plant_name'],
        'crop_type': doc['crop_type'],
        'location': doc.get('location', ''),
        'created_at': doc['created_at'].isoformat(),
        'total_records': total_records,
    }


def create_plant(user_oid, plant_name, crop_type, location=''):
    """Insert a plant and return its document."""
    doc = {
        'user_id': user_oid,
        'plant_id': uuid.uuid4().hex[:10],      # short public code used in URLs
        'plant_name': plant_name[:100],
        'crop_type': crop_type[:50],
        'location': (location or '')[:100],
        'created_at': utcnow(),
    }
    doc['_id'] = get_db().plants.insert_one(doc).inserted_id
    return doc


def get_plant(plant_code, user_oid):
    """Find a plant by its public code - only if it belongs to this user."""
    return get_db().plants.find_one({'plant_id': plant_code, 'user_id': user_oid})


def list_plants(user_oid):
    """All plants of a user (newest first) as template-friendly dicts."""
    d = get_db()
    result = []
    for doc in d.plants.find({'user_id': user_oid}).sort('created_at', DESCENDING):
        total = d.disease_records.count_documents({'plant_ref': doc['_id']})
        result.append(plant_to_dict(doc, total))
    return result


def delete_plant(plant_doc):
    """Delete a plant and all its records. Returns the image file names to remove."""
    d = get_db()
    filenames = [r['image_filename'] for r in
                 d.disease_records.find({'plant_ref': plant_doc['_id']}, {'image_filename': 1})]
    d.disease_records.delete_many({'plant_ref': plant_doc['_id']})
    d.plants.delete_one({'_id': plant_doc['_id']})
    return filenames


# ------------------------------------------------------------------ records
def add_record(plant_doc, image_filename, disease_class, confidence,
               severity_score, affected_area_pct, timestamp, notes=''):
    doc = {
        'plant_ref': plant_doc['_id'],
        'user_id': plant_doc['user_id'],        # denormalised for quick ownership checks
        'image_filename': image_filename,
        'disease_class': disease_class,
        'confidence': float(confidence),
        'severity_score': float(severity_score),
        'affected_area_pct': float(affected_area_pct),
        'timestamp': timestamp,
        'notes': notes or '',
        'created_at': utcnow(),
    }
    return get_db().disease_records.insert_one(doc).inserted_id


def list_records(plant_doc):
    """Raw record documents of a plant, newest observation first."""
    return list(get_db().disease_records
                .find({'plant_ref': plant_doc['_id']})
                .sort('timestamp', DESCENDING))


def get_record(record_id):
    oid = to_object_id(record_id)
    if oid is None:
        return None
    return get_db().disease_records.find_one({'_id': oid})


def delete_record(record_oid):
    get_db().disease_records.delete_one({'_id': record_oid})


def record_to_dict(doc):
    return {
        'id': str(doc['_id']),
        'plant_ref': str(doc['plant_ref']),
        'image_filename': doc['image_filename'],
        'disease_class': doc['disease_class'],
        'confidence': doc['confidence'],
        'severity_score': doc['severity_score'],
        'affected_area_pct': doc['affected_area_pct'],
        'timestamp': doc['timestamp'].isoformat(),
        'notes': doc.get('notes', ''),
    }
