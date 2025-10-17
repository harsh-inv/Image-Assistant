import os
import base64
import time
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Environment variables
AICORE_AUTH_URL = os.getenv('AICORE_AUTH_URL')
AICORE_CLIENT_ID = os.getenv('AICORE_CLIENT_ID')
AICORE_CLIENT_SECRET = os.getenv('AICORE_CLIENT_SECRET')
AICORE_BASE_URL = os.getenv('AICORE_BASE_URL')
AICORE_RESOURCE_GROUP = os.getenv('AICORE_RESOURCE_GROUP')

# Load model
def load_model():
    try:
        proxy_client = get_proxy_client("gen-ai-hub")
        return GenerativeModel(
            deployment_id="d0f921fd2fef0484",
            model_name="gemini-2.0-flash",
            proxy_client=proxy_client
        )
    except Exception as e:
        print(f"Model loading error: {e}")
        return None

model = load_model()

# Session storage
sessions = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/init_session', methods=['POST'])
def init_session():
    data = request.json
    session_id = data.get('session_id')

    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': []
        }

    return jsonify({
        'success': True,
        'files': sessions[session_id]['files'],
        'ticket_counter': sessions[session_id]['ticket_counter']
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')

    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': []
        }

    uploaded_files = []
    files = request.files.getlist('files')

    for file in files:
        if file:
            filename = f"{int(time.time())}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Read file and encode as base64 for preview
            with open(filepath, 'rb') as f:
                file_data = f.read()
                base64_data = base64.b64encode(file_data).decode('utf-8')

                # Determine MIME type
                if filename.lower().endswith('.png'):
                    mime_type = 'image/png'
                elif filename.lower().endswith(('.jpg', '.jpeg')):
                    mime_type = 'image/jpeg'
                elif filename.lower().endswith('.gif'):
                    mime_type = 'image/gif'
                elif filename.lower().endswith('.bmp'):
                    mime_type = 'image/bmp'
                elif filename.lower().endswith('.webp'):
                    mime_type = 'image/webp'
                else:
                    mime_type = 'image/jpeg'  # default

            sessions[session_id]['files'].append({
                'filename': filename,
                'base64': base64_data,
                'mime_type': mime_type
            })

            uploaded_files.append({
                'filename': filename,
                'base64': base64_data,
                'mime_type': mime_type
            })

    return jsonify({
        'success': True,
        'files': uploaded_files
    })

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    session_id = data.get('session_id')
    message = data.get('message')
    is_voice_input = data.get('is_voice_input', False)

    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': []
        }

    try:
        # Add user message to session
        sessions[session_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })

        # Generate response
        if model:
            user_parts = []

            # Add text message
            if message:
                user_parts.append({"text": message})

            # Add uploaded image files in the correct format
            for file_info in sessions[session_id]['files']:
                filename = file_info['filename'] if isinstance(file_info, dict) else file_info
                filepath = os.path.join(UPLOAD_FOLDER, filename)

                if os.path.exists(filepath):
                    # Check if it's an image file
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                        with open(filepath, 'rb') as image_file:
                            image_data = image_file.read()

                        # Encode as base64
                        encoded_image = base64.b64encode(image_data).decode('utf-8')

                        # Determine MIME type based on extension
                        if filename.lower().endswith('.png'):
                            mime_type = 'image/png'
                        elif filename.lower().endswith(('.jpg', '.jpeg')):
                            mime_type = 'image/jpeg'
                        elif filename.lower().endswith('.gif'):
                            mime_type = 'image/gif'
                        elif filename.lower().endswith('.bmp'):
                            mime_type = 'image/bmp'
                        elif filename.lower().endswith('.webp'):
                            mime_type = 'image/webp'

                        # Add image in correct format (image goes first)
                        user_parts.insert(0, {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": encoded_image
                            }
                        })

            # Generate content with properly formatted parts
            response = model.generate_content([
                {"role": "user", "parts": user_parts}
            ])

            bot_response = response.text

            # Check if response contains hazard/risk/broken keywords
            hazard_keywords = [
                'hazard', 'hazards', 'risk', 'risks', 'danger', 'dangerous',
                'broken', 'damaged', 'crack', 'cracked', 'defect', 'defective',
                'unsafe', 'malfunction', 'failure', 'fault', 'faulty',
                'concern', 'issue', 'problem', 'warning', 'alert'
            ]

            show_ticket_button = any(keyword in bot_response.lower() for keyword in hazard_keywords)

            # Add bot message to session
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })

            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'show_ticket_button': show_ticket_button,
                'video': None,
                'video_name': None,
                'session_ended': False
            })
        else:
            return jsonify({
                'error': 'Model not available',
                'response': 'I apologize, but the AI model is currently unavailable.'
            })

    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({
            'error': str(e),
            'response': 'An error occurred while processing your request.'
        })

@app.route('/api/create-ticket', methods=['POST'])
def create_ticket():
    data = request.json
    session_id = data.get('session_id')

    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': []
        }

    # Increment ticket counter
    sessions[session_id]['ticket_counter'] += 1
    ticket_number = f"Q{sessions[session_id]['ticket_counter']:03d}"

    # Create ticket data
    ticket_data = {
        'ticket_number': ticket_number,
        'timestamp': datetime.now().isoformat(),
        'session_id': session_id,
        'type': 'quality_inspection'
    }

    return jsonify({
        'success': True,
        'ticket_number': ticket_number,
        'message': f'Quality Inspection Ticket {ticket_number} created successfully!'
    })

@app.route('/export/json', methods=['POST'])
def export_json():
    data = request.json
    session_id = data.get('session_id')

    if session_id in sessions:
        return jsonify({
            'session_id': session_id,
            'messages': sessions[session_id]['messages'],
            'files': [f['filename'] if isinstance(f, dict) else f for f in sessions[session_id]['files']],
            'ticket_counter': sessions[session_id]['ticket_counter']
        })
    else:
        return jsonify({'error': 'Session not found'})

@app.route('/clear', methods=['POST'])
def clear_chat():
    data = request.json
    session_id = data.get('session_id')

    if session_id in sessions:
        # Delete uploaded files
        for file_info in sessions[session_id]['files']:
            filename = file_info['filename'] if isinstance(file_info, dict) else file_info
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except Exception as e:
                print(f"Error deleting file {filename}: {e}")

        # Clear session data but keep ticket counter
        sessions[session_id]['messages'] = []
        sessions[session_id]['files'] = []

    return jsonify({'success': True})

@app.route('/feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    session_id = data.get('session_id')
    rating = data.get('rating')
    comment = data.get('comment', '')

    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': []
        }

    feedback_entry = {
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }

    sessions[session_id]['feedback'].append(feedback_entry)

    return jsonify({'success': True})

@app.route('/export/feedback', methods=['POST'])
def export_feedback():
    data = request.json
    session_id = data.get('session_id')

    if session_id in sessions and sessions[session_id]['feedback']:
        csv_data = "Timestamp,Rating,Comment\n"
        for fb in sessions[session_id]['feedback']:
            csv_data += f"{fb['timestamp']},{fb['rating']},\"{fb['comment']}\"\n"

        return jsonify({
            'success': True,
            'csv_data': csv_data,
            'filename': f'feedback_{session_id}.csv'
        })
    else:
        return jsonify({
            'success': False,
            'error': 'No feedback data available'
        })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
