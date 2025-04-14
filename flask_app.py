import os
import tempfile
import base64
import sqlite3
import io
import datetime
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderServiceError

# Import the pothole detection function from the existing file
from pothole_detection import run_pothole_detection

from duplication_detection_code import get_duplicate_detector

# (Duplication detection will be added after basic flow is working)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_coordinates_from_address(street, city, state, zipcode):
    """
    Uses geopy to get (latitude, longitude) from address fields.
    Country is always set to India.
    Returns (lat, lon) tuple or (None, None) if not found.
    """
    geolocator = Nominatim(user_agent="pothole-complaint-app")
    address = f"{street}, {city}, {state}, {zipcode}, India"
    try:
        location = geolocator.geocode(address)
        if location:
            return location.latitude, location.longitude
        else:
            return None, None
    except GeocoderServiceError:
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

# Pothole DB setup
POTHOLE_DB = os.path.join(os.path.dirname(__file__), 'pothole_data.db')

def init_pothole_db():
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS pothole_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            input_image BLOB,
            input_filename TEXT,
            detection_result TEXT,
            annotated_image BLOB,
            detected_at TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS pothole_stats (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            total_potholes INTEGER,
            high_priority_count INTEGER,
            medium_priority_count INTEGER,
            low_priority_count INTEGER,
            last_updated TIMESTAMP
        )
    ''')
    # Ensure a single row exists for stats
    c.execute('SELECT COUNT(*) FROM pothole_stats')
    if c.fetchone()[0] == 0:
        c.execute('''
            INSERT INTO pothole_stats (id, total_potholes, high_priority_count, medium_priority_count, low_priority_count, last_updated)
            VALUES (1, 0, 0, 0, 0, ?)
        ''', (datetime.datetime.now(),))
    conn.commit()
    conn.close()

init_pothole_db()

@app.route('/detect_pothole', methods=['POST'])
def detect_pothole():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Run pothole detection
    result_json, annotated_image_bytes = run_pothole_detection(file_path)

    if result_json is None or annotated_image_bytes is None:
        return jsonify({'error': 'Detection failed'}), 500

    # Store in pothole_images table
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('''
        INSERT INTO pothole_images (input_image, input_filename, detection_result, annotated_image, detected_at)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        file.read() if hasattr(file, 'stream') else open(file_path, 'rb').read(),
        filename,
        str(result_json),
        annotated_image_bytes,
        datetime.datetime.now()
    ))

    # Update stats
    # Extract priority info from result_json
    total = result_json.get('total_potholes', 0)
    high = 0
    medium = 0
    low = 0
    # Robustly count priorities
    priorities = result_json.get('individual_priorities')
    if (
        isinstance(priorities, list)
        and len(priorities) == total
        and all(p in ('high', 'medium', 'low') for p in priorities)
    ):
        for p in priorities:
            if p == 'high':
                high += 1
            elif p == 'medium':
                medium += 1
            elif p == 'low':
                low += 1
    elif 'road_priority' in result_json and total > 0:
        # Fallback: if only road_priority is available, assign all to that priority
        if result_json['road_priority'] == 'high':
            high = total
        elif result_json['road_priority'] == 'medium':
            medium = total
        elif result_json['road_priority'] == 'low':
            low = total
    # If priorities are missing or malformed, only update total, not priority counts

    # Update stats row
    c.execute('SELECT total_potholes, high_priority_count, medium_priority_count, low_priority_count FROM pothole_stats WHERE id=1')
    stats = c.fetchone()
    if stats:
        new_total = stats[0] + total
        new_high = stats[1] + high
        new_medium = stats[2] + medium
        new_low = stats[3] + low
        c.execute('''
            UPDATE pothole_stats
            SET total_potholes=?, high_priority_count=?, medium_priority_count=?, low_priority_count=?, last_updated=?
            WHERE id=1
        ''', (new_total, new_high, new_medium, new_low, datetime.datetime.now()))
    conn.commit()
    conn.close()

    # Encode annotated image as base64 for frontend display
    annotated_image_b64 = base64.b64encode(annotated_image_bytes).decode('utf-8')

    # Clean up uploaded file
    try:
        os.remove(file_path)
    except Exception:
        pass

    return jsonify({
        'result': result_json,
        'annotated_image_b64': annotated_image_b64
    })

import sqlite3
import io
import datetime
from duplication_detection_code import get_duplicate_detector

# Serve static files (CSS, JS)
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# Complaints DB setup
COMPLAINTS_DB = os.path.join(os.path.dirname(__file__), 'complaints.db')

def init_complaints_db():
    conn = sqlite3.connect(COMPLAINTS_DB)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS complaints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            location_lat REAL,
            location_lon REAL,
            issue_type TEXT,
            image BLOB,
            image_filename TEXT,
            submitted_at TIMESTAMP,
            is_duplicate INTEGER,
            original_report_id INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_complaints_db()

@app.route('/raise_complaint', methods=['POST'])
def raise_complaint():
    # Get form fields
    text = request.form.get('text')
    issue_type = request.form.get('issue_type')
    street = request.form.get('street')
    city = request.form.get('city')
    state = request.form.get('state')
    zipcode = request.form.get('zipcode')
    image_file = request.files.get('image')

    if not all([text, issue_type, street, city, state, zipcode, image_file]):
        return jsonify({'error': 'All fields are required.'}), 400

    # Get coordinates from address
    lat, lon = get_coordinates_from_address(street, city, state, zipcode)
    if lat is None or lon is None:
        return jsonify({'error': 'Could not geocode the provided address.'}), 400

    # Read image bytes
    image_bytes = image_file.read()
    image_filename = secure_filename(image_file.filename)

    # Prepare report dict for duplication detection
    report = {
        'text': text,
        'location': (lat, lon),
        'issue_type': issue_type,
        'image_bytes': image_bytes
    }

    # Load all existing complaints for duplication detection
    conn = sqlite3.connect(COMPLAINTS_DB)
    c = conn.cursor()
    c.execute('SELECT id, text, location_lat, location_lon, issue_type, image FROM complaints WHERE is_duplicate=0')
    rows = c.fetchall()
    detector = get_duplicate_detector()
    for row in rows:
        db_report = {
            'id': row[0],
            'text': row[1],
            'location': (row[2], row[3]),
            'issue_type': row[4],
            'image_bytes': row[5]
        }
        detector.add_report(db_report)

    # Check for duplicates
    is_duplicate, similar_reports, confidence = detector.find_duplicates(report)
    if is_duplicate:
        original_id = similar_reports[0] if similar_reports else None
        # Store as duplicate for record-keeping
        c.execute('''
            INSERT INTO complaints (text, location_lat, location_lon, issue_type, image, image_filename, submitted_at, is_duplicate, original_report_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (text, lat, lon, issue_type, image_bytes, image_filename, datetime.datetime.now(), 1, original_id))
        conn.commit()
        conn.close()
        return jsonify({'message': f'Duplicate complaint detected. Similar to complaint ID {original_id}.'}), 200

    # Not duplicate, store as new complaint
    c.execute('''
        INSERT INTO complaints (text, location_lat, location_lon, issue_type, image, image_filename, submitted_at, is_duplicate, original_report_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (text, lat, lon, issue_type, image_bytes, image_filename, datetime.datetime.now(), 0, None))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Complaint registered successfully.'}), 200

@app.route('/pothole_stats', methods=['GET'])
def pothole_stats():
    conn = sqlite3.connect(POTHOLE_DB)
    c = conn.cursor()
    c.execute('SELECT total_potholes, high_priority_count, medium_priority_count, low_priority_count, last_updated FROM pothole_stats WHERE id=1')
    row = c.fetchone()
    conn.close()
    if row:
        return jsonify({
            'total_potholes': row[0],
            'high_priority_count': row[1],
            'medium_priority_count': row[2],
            'low_priority_count': row[3],
            'last_updated': row[4]
        })
    else:
        return jsonify({'error': 'Stats not found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
