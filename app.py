import os
import base64
import time
import json
import random
from datetime import datetime
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
from gen_ai_hub.proxy.native.google_vertexai.clients import GenerativeModel
from gen_ai_hub.proxy.core.proxy_clients import get_proxy_client
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
import html

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

def get_system_prompt(file_type):
    """Generate system prompt based on file type"""
    
    base_prompt = """You are a professional Image/Audio Analysis Assistant for quality inspection and safety assessment.

CRITICAL CONVERSATION RULES:
1. When user acknowledges (says "ok", "nice", "thanks", "good", "no", "yes" etc.) - Give a SINGLE SHORT sentence response
2. NEVER repeat previous analysis unless explicitly asked with words like "again", "repeat", "show me"  
3. If user says "no" after you asked if they want more info - Respond: "Understood. Let me know if you need anything else."
4. Only provide detailed analysis for NEW questions or when user uploads NEW files
5. For follow-up questions, answer concisely based on previous analysis

RESPONSE LENGTH:
- Acknowledgments: 1 sentence maximum
- Follow-up questions: 2-3 sentences
- New analysis requests: Full structured response"""

    if file_type == 'image':
        return base_prompt + """

IMAGE ANALYSIS STRUCTURE (only for NEW analysis requests):

1. **Overall Impression:** Brief summary

2. **Key Observations:** 
   - Notable features (3-5 bullet points)

3. **Quality Assessment:** 
   - Defects, damage, irregularities
   - **Highlight: hazards, risks, dangers** using these exact words
   - Material condition

4. **Potential Hazards:** (if any)
   - Safety concerns with severity
   - Fire, electrical, structural issues
   
5. **Recommendations:** Actions needed

Critical terms to use when applicable: hazard, risk, danger, damaged, broken, defect, unsafe, malfunction, failure.
"""
    elif file_type == 'audio':
        return base_prompt + """

AUDIO ANALYSIS STRUCTURE (only for NEW analysis requests):

1. **Overall Impression:** Brief description

2. **Key Observations:**
   - Content type and quality
   - Notable characteristics

3. **Sentiment/Tone:** (for speech)
   - Emotional tone
   - Speaker characteristics

4. **Technical Quality:**
   - Audio clarity
   - Background noise
   - Issues or distortions

5. **Key Insights:** Main takeaways
"""
    else:
        return base_prompt + """

GENERAL ANALYSIS:
Provide structured analysis only when explicitly requested.
For acknowledgments, respond with 1 sentence maximum.
"""

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
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0
        }
    
    return jsonify({
        'success': True,
        'files': sessions[session_id]['files'],
        'ticket_counter': sessions[session_id]['ticket_counter'],
        'ticket_created': sessions[session_id]['ticket_created'],
        'feedback_submitted': sessions[session_id]['feedback_submitted'],
        'ticket_button_clicked': sessions[session_id].get('ticket_button_clicked', False)
    })

@app.route('/upload', methods=['POST'])
def upload_file():
    session_id = request.form.get('session_id')
    if session_id not in sessions:
        sessions[session_id] = {
            'messages': [],
            'files': [],
            'ticket_counter': 0,
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0
        }
    
    # Clear existing files (only one file at a time)
    for file_info in sessions[session_id]['files']:
        filename = file_info['filename'] if isinstance(file_info, dict) else file_info
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file {filename}: {e}")
    
    sessions[session_id]['files'] = []
    
    # IMPORTANT: Reset ticket button state when new file is uploaded
    sessions[session_id]['ticket_button_clicked'] = False
    sessions[session_id]['ticket_created'] = False
    sessions[session_id]['last_analysis'] = None
    sessions[session_id]['awaiting_followup'] = False
    sessions[session_id]['consecutive_no_count'] = 0
    
    uploaded_files = []
    files = request.files.getlist('files')
    
    # Only process the first file (single file upload)
    if files and files[0]:
        file = files[0]
        filename = f"{int(time.time())}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Read file and encode as base64 for preview
        with open(filepath, 'rb') as f:
            file_data = f.read()
        base64_data = base64.b64encode(file_data).decode('utf-8')
        
        # Determine MIME type for images
        if filename.lower().endswith('.png'):
            mimetype = 'image/png'
        elif filename.lower().endswith(('.jpg', '.jpeg')):
            mimetype = 'image/jpeg'
        elif filename.lower().endswith('.gif'):
            mimetype = 'image/gif'
        elif filename.lower().endswith('.bmp'):
            mimetype = 'image/bmp'
        elif filename.lower().endswith('.webp'):
            mimetype = 'image/webp'
        # Audio file MIME types
        elif filename.lower().endswith('.wav'):
            mimetype = 'audio/wav'
        elif filename.lower().endswith('.mp3'):
            mimetype = 'audio/mp3'
        elif filename.lower().endswith('.aiff'):
            mimetype = 'audio/aiff'
        elif filename.lower().endswith('.aac'):
            mimetype = 'audio/aac'
        elif filename.lower().endswith('.ogg'):
            mimetype = 'audio/ogg'
        elif filename.lower().endswith('.flac'):
            mimetype = 'audio/flac'
        else:
            mimetype = 'application/octet-stream'
        
        sessions[session_id]['files'].append({
            'filename': filename,
            'base64': base64_data,
            'mimetype': mimetype
        })
        
        uploaded_files.append({
            'filename': filename,
            'base64': base64_data,
            'mimetype': mimetype
        })
    
    # Update last interaction time and store upload completion time
    upload_time = time.time()
    sessions[session_id]['last_interaction'] = upload_time
    sessions[session_id]['upload_completed_time'] = upload_time  # NEW: Store upload time
    
    return jsonify({
        'success': True,
        'files': uploaded_files,
        'ticket_button_clicked': sessions[session_id]['ticket_button_clicked'],
        'ticket_created': sessions[session_id]['ticket_created'],
        'upload_completed_time': upload_time  # NEW: Send to frontend
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
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False,
            'consecutive_no_count': 0
        }
    
    # Update last interaction time
    sessions[session_id]['last_interaction'] = time.time()
    
    try:
        # Add user message to session
        sessions[session_id]['messages'].append({
            'role': 'user',
            'content': message,
            'timestamp': datetime.now().isoformat()
        })
        
        # Normalize message
        normalized_message = message.lower().strip().replace("'", "").replace(",", "").replace(".", "")
        
        # Check if this is a negative/dismissive response
        negative_responses = ['no', 'nope', 'nah', 'not needed', 'no need', 'no thanks', 
                             'not really', 'im good', "i'm good", 'all good', 'thats all', 
                             "that's all", 'nothing else', 'nothing more']
        
        is_negative = any(neg in normalized_message for neg in negative_responses)
        
        # Track consecutive "no" responses
        if is_negative:
            sessions[session_id]['consecutive_no_count'] = sessions[session_id].get('consecutive_no_count', 0) + 1
        else:
            sessions[session_id]['consecutive_no_count'] = 0
        
        # If user has said "no" 2 or more times consecutively, end the session
        if sessions[session_id]['consecutive_no_count'] >= 2:
            bot_response = "Thank you for using Image/Audio Assistant! Have a great day!"
            
            sessions[session_id]['messages'].append({
                'role': 'assistant',
                'content': bot_response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Reset consecutive count
            sessions[session_id]['consecutive_no_count'] = 0
            
            return jsonify({
                'success': True,
                'response': bot_response,
                'is_voice_input': is_voice_input,
                'show_ticket_button': False,
                'ticket_created': sessions[session_id]['ticket_created'],
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'ticket_button_clicked': sessions[session_id]['ticket_button_clicked'],
                'video': None,
                'video_name': None,
                'session_ended': True,
                'trigger_feedback': True
            })
        
        # Generate response
        if model:
            user_parts = []
            
            # Determine file type from uploaded files
            file_type = None
            has_image_file = False
            
            for file_info in sessions[session_id]['files']:
                filename = file_info['filename'] if isinstance(file_info, dict) else file_info
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                    file_type = 'image'
                    has_image_file = True
                    break
                elif filename.lower().endswith(('.wav', '.mp3', '.aiff', '.aac', '.ogg', '.flac')):
                    file_type = 'audio'
                    break
            
            # UNIFIED INTELLIGENT SYSTEM PROMPT - adapts to any image type
            if file_type == 'image':
                system_prompt = """You are a professional Image Analysis Assistant that intelligently adapts to any type of image.

CRITICAL ANALYSIS RULES:

1. **First, examine the image** to understand what it actually shows
2. **Adapt your analysis format** based on the image content
3. **Never state what the image is NOT** - just analyze what it IS
4. **Be specific with numbers, measurements, and details**

---

IF IMAGE SHOWS: Business Dashboard, Charts, Data Visualizations, Analytics Reports, KPI Displays
THEN PROVIDE:

**Overview**: Brief description of what this dashboard/visualization shows

**Key Metrics Analysis**: 
- Identify each metric with its actual value
- Explain what each number/percentage represents
- Note any standout figures

**Performance Insights**:
- Areas of strong performance
- Areas of concern or underperformance
- Budget variances (over/under)
- Growth trends (Year-over-Year, Quarter-over-Quarter)

**Comparative Analysis**:
- Regional/Product/Customer comparisons
- Top and bottom performers
- Segment-specific insights

**Business Interpretation**: Explain in simple, plain language what the data means

**Key Takeaways**: List 3-5 most important insights

---

IF IMAGE SHOWS: Physical Objects, Equipment, Machinery, Electrical Components, Buildings, Infrastructure, or Defects
THEN PROVIDE:

**Overall Assessment**: What the image shows and general condition

**Key Observations**:
- Notable features, components, or concerns
- Visible defects, damage, or irregularities
- Material condition and state

**Quality/Safety Assessment**:
- **Defects Identified**: Type, location, severity, potential causes
- **Hazards Present**: Fire risk, electrical issues, structural problems, safety concerns
- **Risk Level**: Assessment of danger or risk
- Material degradation, corrosion, wear, or failure signs

**Potential Hazards** (if any):
- Safety concerns with specific severity
- Fire, electrical, structural, or operational risks
- Immediate dangers

**Recommendations**:
- Immediate actions required
- Safety precautions needed
- Maintenance or repairs suggested
- Whether quality inspection ticket should be raised
- Further inspection areas

**CRITICAL TERMS TO USE WHEN APPLICABLE**: hazard, risk, danger, damaged, broken, defect, unsafe, malfunction, failure, crack, corrosion, wear, fault, faulty

---

RESPONSE LENGTH RULES:
- Acknowledgments (ok, thanks, nice, good, etc.): 1 sentence maximum
- Follow-up questions about previous analysis: 2-3 sentences
- New analysis requests: Full structured response as shown above
- **Never repeat previous analysis** unless explicitly asked with words like "again", "repeat", "show me"

CONVERSATION CONTEXT:
- If user just acknowledged (said "ok", "thanks", etc.), respond briefly: "You're welcome! Let me know if you need anything else."
- If user said "no" after being asked if they want more info: "Understood. Feel free to ask if you need anything else."
- Only provide detailed analysis for NEW questions or when user uploads NEW files

Current user question: {message}

Analyze the image and provide the appropriate detailed response."""

            elif file_type == 'audio':
                system_prompt = get_system_prompt(file_type)
            else:
                system_prompt = get_system_prompt(None)
            
            # Check if this is an acknowledgment
            acknowledgments = [
                'ok', 'okay', 'okey', 'oke', 'k',
                'nice', 'good', 'great', 'excellent', 'awesome', 'perfect', 'cool', 'fine',
                'thanks', 'thank you', 'thankyou', 'thx', 'ty',
                'alright', 'got it', 'understood', 'i see', 'i understand',
                'yes', 'yeah', 'yep', 'yup', 'sure', 'of course'
            ]
            
            # Add negative responses to acknowledgments for brief handling
            acknowledgments.extend(negative_responses)
            
            is_acknowledgment = (
                normalized_message in acknowledgments or
                (len(normalized_message.split()) <= 3 and any(ack in normalized_message for ack in acknowledgments))
            ) and not any(question_word in normalized_message for question_word in [
                'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could', 
                'would', 'should', 'is', 'are', 'does', 'do', 'analyze', 'explain', 
                'tell', 'show', 'describe'
            ])
            
            # Check relevance for non-acknowledgment questions when files are uploaded
            if not is_acknowledgment and sessions[session_id]['files'] and file_type:
                # Quick relevance check with the AI
                relevance_check_prompt = f"""You are analyzing whether a user's question is relevant to {file_type} analysis.

The user has uploaded a {file_type} file and is asking: "{message}"

Respond with ONLY one word:
- "RELEVANT" if the question is about analyzing, understanding, or discussing the uploaded {file_type}
- "IRRELEVANT" if the question is completely unrelated to the {file_type} (like asking about weather, time, unrelated topics, etc.)

Request ID: {random.randint(1000, 9999)}
Your response (one word only):"""

                relevance_response = model.generate_content([
                    {"role": "user", "parts": [{"text": relevance_check_prompt}]}
                ])
                
                relevance_result = relevance_response.text.strip().upper()
                
                # If question is irrelevant, return a polite redirect
                if "IRRELEVANT" in relevance_result:
                    if file_type == 'image':
                        bot_response = "I can help you analyze the image you've uploaded. It seems your question isn't directly related to the image. Could you please ask a question about the image, or upload a new file if you'd like me to analyze something different?"
                    elif file_type == 'audio':
                        bot_response = "I can help you analyze the audio you've uploaded. It seems your question isn't directly related to the audio. Could you please ask a question about the audio, or upload a new file if you'd like me to analyze something different?"
                    else:
                        bot_response = "I can help you with your file analysis. It seems your question isn't directly related to the uploaded file. Could you please ask a question about the file, or upload a new one if you'd like me to analyze something different?"
                    
                    sessions[session_id]['messages'].append({
                        'role': 'assistant',
                        'content': bot_response,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    return jsonify({
                        'success': True,
                        'response': bot_response,
                        'is_voice_input': is_voice_input,
                        'show_ticket_button': False,
                        'ticket_created': sessions[session_id]['ticket_created'],
                        'feedback_submitted': sessions[session_id]['feedback_submitted'],
                        'ticket_button_clicked': sessions[session_id]['ticket_button_clicked'],
                        'video': None,
                        'video_name': None,
                        'session_ended': False
                    })
            
            # Build context
            if is_acknowledgment and sessions[session_id]['last_analysis']:
                # For acknowledgments, give a brief response without re-analyzing
                context_message = f"""The user previously received a detailed analysis. User just responded with: "{message}"

This is just an acknowledgment, NOT a request for new analysis.

Respond VERY BRIEFLY with ONE of these options:
- If they said "ok/nice/good/thanks": "You're welcome! Feel free to ask if you need anything else or upload a new file for analysis."
- If they said "no/no need/not needed": "Understood. Let me know if you need anything else."
- If they said "yes": "What specific aspect would you like me to elaborate on?"

Do NOT repeat the analysis. Keep response to 1-2 sentences maximum."""
                user_parts.append({"text": context_message})
                
                # Don't add files for acknowledgment responses to save processing
            else:
                # For actual questions, include system prompt and conversation context
                context_message = system_prompt
                
                # Add recent conversation history for context (last 2 exchanges)
                recent_messages = sessions[session_id]['messages'][-4:] if len(sessions[session_id]['messages']) > 4 else sessions[session_id]['messages']
                if len(recent_messages) > 1:  # If there's conversation history
                    context_message += "\n\nRECENT CONVERSATION CONTEXT:\n"
                    for msg in recent_messages[:-1]:  # Exclude current message
                        role = "User" if msg['role'] == 'user' else "Assistant"
                        context_message += f"{role}: {msg['content'][:200]}...\n"
                
                context_message += f"\n\nCurrent user message: {message}"
                user_parts.append({"text": context_message})
                
                # Add uploaded files (images or audio) in the correct format
                for file_info in sessions[session_id]['files']:
                    filename = file_info['filename'] if isinstance(file_info, dict) else file_info
                    filepath = os.path.join(UPLOAD_FOLDER, filename)
                    
                    if os.path.exists(filepath):
                        # Check if it's an image file
                        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp')):
                            with open(filepath, 'rb') as image_file:
                                image_data = image_file.read()
                                encoded_image = base64.b64encode(image_data).decode('utf-8')
                            
                            # Determine image MIME type
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
                            
                            # Add image
                            user_parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded_image
                                }
                            })
                        
                        # Check if it's an audio file
                        elif filename.lower().endswith(('.wav', '.mp3', '.aiff', '.aac', '.ogg', '.flac')):
                            with open(filepath, 'rb') as audio_file:
                                audio_data = audio_file.read()
                                encoded_audio = base64.b64encode(audio_data).decode('utf-8')
                            
                            # Determine audio MIME type
                            if filename.lower().endswith('.wav'):
                                mime_type = 'audio/wav'
                            elif filename.lower().endswith('.mp3'):
                                mime_type = 'audio/mp3'
                            elif filename.lower().endswith('.aiff'):
                                mime_type = 'audio/aiff'
                            elif filename.lower().endswith('.aac'):
                                mime_type = 'audio/aac'
                            elif filename.lower().endswith('.ogg'):
                                mime_type = 'audio/ogg'
                            elif filename.lower().endswith('.flac'):
                                mime_type = 'audio/flac'
                            
                            # Add audio
                            user_parts.append({
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": encoded_audio
                                }
                            })
            
            # Generate content with properly formatted parts
            response = model.generate_content([
                {"role": "user", "parts": user_parts}
            ])
            
            bot_response = response.text
            
            # Store this as last analysis if it's not an acknowledgment response
            if not is_acknowledgment:
                sessions[session_id]['last_analysis'] = bot_response
                sessions[session_id]['awaiting_followup'] = True
            else:
                sessions[session_id]['awaiting_followup'] = False
            
            # Check if response contains hazard/risk/broken keywords for DEFECTS
            hazard_keywords = [
                'hazard', 'hazards', 'risk', 'risks', 'danger', 'dangerous',
                'broken', 'damaged', 'crack', 'cracked', 'defect', 'defective',
                'unsafe', 'malfunction', 'failure', 'fault', 'faulty',
                'concern', 'issue', 'problem', 'warning', 'alert'
            ]

            # Dashboard/business context keywords - EXPANDED
            dashboard_context_keywords = [
                'dashboard', 'chart', 'graph', 'visualization', 'metric', 'metrics',
                'kpi', 'analytics', 'report', 'data', 'statistics', 'performance',
                'trend', 'bar chart', 'pie chart', 'line graph', 'scatter plot',
                'heatmap', 'infographic', 'scorecard', 'diagram', 'table',
                'financial', 'budget', 'revenue', 'sales', 'quarterly', 'year-over-year',
                'yoy', 'ytd', 'delta', 'variance', 'actuals', 'forecast',
                'business', 'customer', 'product', 'region', 'growth',
                'development', 'quarter', 'q1', 'q2', 'q3', 'q4',
                'overview', 'key metrics', 'performance insights', 'comparative analysis',
                'business interpretation', 'key takeaways'
            ]

            # Financial/business indicators in response
            financial_indicators = [
                'budget', 'quarter', 'yoy', 'ytd', 'revenue', 'sales',
                'financial', 'growth %', 'actuals', 'forecast', 'variance',
                'delta to budget', 'year-over-year', 'year-to-date',
                'powerstick co', 'zato service', 'energy supply', 'healthyfood',
                'municipality', 'oblasecurity', 'health & hygiene', 'sweet gmbh',
                'paper & towel', 'coolcar ag', 'cs mee', 'cs north america',
                'cs emea', 'cs apj', 'cs latin america', 'cs g.china',
                'epm', 'bi & predictive', 'customer', 'top 10', 'acv ranges'
            ]

            # Check if it's a dashboard/business context
            is_dashboard_context = any(keyword in bot_response.lower() for keyword in dashboard_context_keywords)

            # Additional check for financial context
            is_financial_context = any(indicator in bot_response.lower() for indicator in financial_indicators)

            # Check if hazard keywords are present
            has_hazard_keywords = any(keyword in bot_response.lower() for keyword in hazard_keywords)

            # Show ticket button ONLY if:
            # 1. Has image file
            # 2. Ticket button not already clicked
            # 3. Response contains hazard/defect keywords
            # 4. NOT in dashboard/financial context
            # 5. NOT an acknowledgment
            show_ticket_button = (
                has_image_file and 
                (not sessions[session_id]['ticket_button_clicked']) and 
                has_hazard_keywords and
                not is_dashboard_context and
                not is_financial_context and
                not is_acknowledgment
            )
            
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
                'ticket_created': sessions[session_id]['ticket_created'],
                'feedback_submitted': sessions[session_id]['feedback_submitted'],
                'ticket_button_clicked': sessions[session_id]['ticket_button_clicked'],
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
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False
        }
    
    # Mark ticket as created and button as clicked for this session
    sessions[session_id]['ticket_created'] = True
    sessions[session_id]['ticket_button_clicked'] = True
    
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
    
    # Update last interaction time
    sessions[session_id]['last_interaction'] = time.time()
    
    return jsonify({
        'success': True,
        'ticket_number': ticket_number,
        'message': f'Quality Inspection Ticket {ticket_number} created successfully!',
        'ticket_created': True,
        'ticket_button_clicked': True
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

@app.route('/export/pdf', methods=['POST'])
def export_pdf():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id not in sessions or not sessions[session_id]['messages']:
        return jsonify({'error': 'No chat history found'}), 404
    
    try:
        # Create PDF filename
        pdf_filename = f'chat_export_{session_id}_{int(time.time())}.pdf'
        pdf_path = os.path.join(STATIC_FOLDER, pdf_filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=16,
            textColor='#2c3e50',
            spaceAfter=30
        )
        
        user_style = ParagraphStyle(
            'UserMessage',
            parent=styles['Normal'],
            fontSize=11,
            textColor='#2980b9',
            leftIndent=20,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        )
        
        bot_style = ParagraphStyle(
            'BotMessage',
            parent=styles['Normal'],
            fontSize=10,
            textColor='#34495e',
            leftIndent=20,
            spaceAfter=15
        )
        
        # Build PDF content
        story = []
        
        # Title
        story.append(Paragraph("Image/Audio Assistant - Chat Export", title_style))
        story.append(Spacer(1, 0.2 * inch))
        
        # Add messages
        for msg in sessions[session_id]['messages']:
            if msg['role'] == 'user':
                story.append(Paragraph(f"<b>You:</b> {html.escape(msg['content'])}", user_style))
            else:
                # Clean bot response for PDF
                content = msg['content'].replace('**', '')
                story.append(Paragraph(f"<b>Assistant:</b> {html.escape(content)}", bot_style))
        
        # Build PDF
        doc.build(story)
        
        # Read PDF file
        with open(pdf_path, 'rb') as f:
            pdf_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Clean up
        os.remove(pdf_path)
        
        return jsonify({
            'success': True,
            'pdf_data': pdf_data,
            'filename': pdf_filename
        })
    
    except Exception as e:
        print(f"PDF export error: {e}")
        return jsonify({'error': str(e)}), 500

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
        sessions[session_id]['ticket_created'] = False
        sessions[session_id]['ticket_button_clicked'] = False
        sessions[session_id]['last_interaction'] = time.time()
        sessions[session_id]['last_analysis'] = None
        sessions[session_id]['awaiting_followup'] = False
        sessions[session_id]['consecutive_no_count'] = 0
        
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
            'feedback': [],
            'ticket_created': False,
            'last_interaction': time.time(),
            'feedback_submitted': False,
            'ticket_button_clicked': False,
            'last_analysis': None,
            'awaiting_followup': False
        }
    
    feedback_entry = {
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.now().isoformat()
    }
    
    sessions[session_id]['feedback'].append(feedback_entry)
    sessions[session_id]['feedback_submitted'] = True
    sessions[session_id]['last_interaction'] = time.time()
    
    return jsonify({
        'success': True,
        'feedback_submitted': True
    })

@app.route('/check_idle', methods=['POST'])
def check_idle():
    data = request.json
    session_id = data.get('session_id')
    
    if session_id in sessions:
        current_time = time.time()
        last_interaction = sessions[session_id]['last_interaction']
        upload_completed_time = sessions[session_id].get('upload_completed_time')
        
        # Calculate idle time
        idle_time = current_time - last_interaction
        
        # If file was recently uploaded, use 10 second threshold
        # Otherwise use 7 second threshold
        if upload_completed_time and (current_time - upload_completed_time) < 15:
            # Within 15 seconds of upload, use 10 second threshold
            idle_threshold = 10
        else:
            # Normal idle threshold
            idle_threshold = 7
            # Clear the upload_completed_time after threshold period
            if upload_completed_time:
                sessions[session_id]['upload_completed_time'] = None
        
        if idle_time >= idle_threshold:
            return jsonify({
                'is_idle': True,
                'idle_time': idle_time
            })
        else:
            return jsonify({
                'is_idle': False,
                'idle_time': idle_time
            })
    else:
        return jsonify({
            'is_idle': False,
            'idle_time': 0
        })

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
