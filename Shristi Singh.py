import streamlit as st
import json
import uuid
from datetime import datetime
import base64
import requests
import time
import io
import cv2
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="AI Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .sidebar .sidebar-content {
        background-color: #202123;
        color: white;
    }
    .user-message {
        background: #1a73e8;
        color: white;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 10px 0;
        margin-left: 10%;
        max-width: 85%;
        word-wrap: break-word;
    }
    .ai-message {
        background: #f0f0f0;
        color: #333;
        padding: 16px 20px;
        border-radius: 8px;
        margin: 10px 0;
        margin-right: 10%;
        max-width: 85%;
        word-wrap: break-word;
        border: 1px solid #e5e5e5;
    }
    .typing-indicator {
        display: inline-block;
        margin-left: 10px;
    }
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #666;
        margin: 0 2px;
        animation: typing 1.4s infinite;
    }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-5px); }
    }
    .confidence-badge {
        background: #4CAF50;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .model-badge {
        background: #2196F3;
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8em;
        margin-left: 10px;
    }
    .ocr-upload-box {
        background: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 10px 0;
    }
    .success-box {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .chat-input-container {
        position: relative;
        margin-top: 20px;
    }
    .upload-button {
        position: absolute;
        right: 100px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 10;
    }
    .image-preview {
        background: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
    st.session_state.current_session = str(uuid.uuid4())
    st.session_state.chat_sessions[st.session_state.current_session] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }

if "processing" not in st.session_state:
    st.session_state.processing = False

if "ollama_models" not in st.session_state:
    st.session_state.ollama_models = ["llama2", "mistral", "codellama"]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama2"

if "ocr_extracted_text" not in st.session_state:
    st.session_state.ocr_extracted_text = ""

if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

if "show_ocr_preview" not in st.session_state:
    st.session_state.show_ocr_preview = False

# Try to import EasyOCR
try:
    import easyocr
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Initialize EasyOCR reader in session state
if OCR_AVAILABLE and "easyocr_reader" not in st.session_state:
    try:
        with st.spinner("🔄 Loading OCR engine... This may take a minute on first run."):
            st.session_state.easyocr_reader = easyocr.Reader(['en'])
    except Exception as e:
        st.error(f"Failed to initialize EasyOCR: {e}")
        OCR_AVAILABLE = False

class SimpleOCRProcessor:
    def extract_text_simple(self, image):
        """Simple text extraction using EasyOCR"""
        try:
            if "easyocr_reader" not in st.session_state:
                st.session_state.easyocr_reader = easyocr.Reader(['en'])
            
            # Convert PIL image to bytes
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='PNG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Extract text using EasyOCR
            results = st.session_state.easyocr_reader.readtext(img_byte_arr)
            
            # Combine all detected text
            extracted_text = ""
            confidence_scores = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Only keep confident detections
                    extracted_text += text + "\n"
                    confidence_scores.append(confidence)
            
            cleaned_text = extracted_text.strip()
            
            if cleaned_text:
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
                return cleaned_text, avg_confidence, f"Found {len(results)} text regions"
            else:
                return "No readable text found in the image.", 0, "No text detected"
                
        except Exception as e:
            return f"OCR Error: {str(e)}", 0, f"Error: {str(e)}"

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
    
    def check_connection(self):
        """Check if Ollama server is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'] for model in models_data.get('models', [])]
            return ["llama2"]  # Default fallback
        except:
            return ["llama2"]  # Default fallback
    
    def generate_response(self, prompt, model=None, context_history=None):
        """Generate response from Ollama"""
        if model is None:
            model = st.session_state.selected_model
            
        url = f"{self.base_url}/api/generate"
        
        # Build context from conversation history
        if context_history:
            context_prompt = self._build_context_prompt(context_history, prompt)
        else:
            context_prompt = prompt
        
        # Simple data structure that works reliably
        data = {
            "model": model,
            "prompt": context_prompt,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=data, timeout=120)  # Increased timeout
            response.raise_for_status()
            result = response.json()
            return result.get('response', 'No response generated'), "high"
            
        except requests.exceptions.ConnectionError:
            return "❌ **Connection Error**: Could not connect to Ollama server. Please make sure Ollama is running with `ollama serve`.", "low"
        except requests.exceptions.Timeout:
            return "❌ **Timeout Error**: Ollama is taking too long to respond. The model might be processing a large request. Try a simpler question.", "low"
        except requests.exceptions.HTTPError as e:
            if response.status_code == 500:
                return "❌ **Server Error**: Ollama encountered an internal error. This often happens when:\n\n1. The model is busy\n2. The prompt is too complex\n3. There's not enough memory\n\nTry asking a simpler question or wait a moment.", "low"
            else:
                return f"❌ **HTTP Error {response.status_code}**: {str(e)}", "low"
        except Exception as e:
            return f"❌ **Unexpected Error**: {str(e)}", "low"
    
    def _build_context_prompt(self, history, current_prompt):
        """Build context from conversation history"""
        # Keep it simple - don't overload with too much context
        if not history:
            return current_prompt
            
        context = "Previous conversation:\n"
        for msg in history[-3:]:  # Only last 3 messages to avoid overload
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content']}\n"
        
        context += f"\nCurrent question: {current_prompt}\nAssistant:"
        return context

def get_ai_response(question, context_history=None):
    """Get response from Ollama with fallback to knowledge base"""
    ollama_client = OllamaClient()
    
    # First try Ollama
    if ollama_client.check_connection():
        response, confidence = ollama_client.generate_response(question, context_history=context_history)
        return response, confidence, "ollama"
    else:
        # Fallback to static knowledge base
        return "🤖 **Ollama Connection Issue**\n\nI can see Ollama is installed but there might be a connection issue. Try:\n\n1. Make sure `ollama serve` is running\n2. Check if the model is loaded: `ollama list`\n3. Try a simple test: `ollama run llama2 'Hello'`\n\nYour OCR text was extracted successfully! The text is ready for analysis once Ollama is connected properly.", "medium", "knowledge_base"

def display_confidence_badge(confidence, source):
    """Display confidence level badge"""
    color_map = {
        "high": "#4CAF50",
        "medium": "#FF9800",
        "low": "#F44336"
    }
    source_map = {
        "ollama": "🤖 AI",
        "knowledge_base": "📚 Knowledge Base"
    }
    return f'''
    <span class="confidence-badge" style="background: {color_map[confidence]};">Confidence: {confidence}</span>
    <span class="model-badge">Source: {source_map[source]}</span>
    '''

def create_new_chat_session():
    """Create a new chat session"""
    new_session_id = str(uuid.uuid4())
    st.session_state.chat_sessions[new_session_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_session = new_session_id
    st.session_state.processing = False
    st.session_state.ocr_extracted_text = ""
    st.session_state.uploaded_image = None
    st.session_state.show_ocr_preview = False
    return new_session_id

def process_uploaded_image(uploaded_file):
    """Process uploaded image and extract text"""
    try:
        # Initialize OCR processor
        processor = SimpleOCRProcessor()
        image = Image.open(uploaded_file)
        
        # Store the uploaded image
        st.session_state.uploaded_image = image
        
        # Extract text
        extracted_text, confidence, info = processor.extract_text_simple(image)
        
        if extracted_text and "No readable text" not in extracted_text:
            st.session_state.ocr_extracted_text = extracted_text
            return True, extracted_text, confidence, info
        else:
            st.session_state.ocr_extracted_text = ""
            return False, extracted_text, confidence, info
            
    except Exception as e:
        st.session_state.ocr_extracted_text = ""
        return False, f"OCR Error: {str(e)}", 0, f"Error: {str(e)}"

def main():
    # Initialize clients
    ollama_client = OllamaClient()
    
    # Sidebar
    with st.sidebar:
        st.title("💬 Chat Settings")
        
        # OCR Status
        if OCR_AVAILABLE:
            st.success("✅ OCR Available")
        else:
            st.warning("📷 OCR Not Available")
            st.markdown("""
            **Install OCR:**
            ```bash
            pip install easyocr opencv-python
            ```
            **Then restart the app**
            """)
        
        # Ollama Status
        st.subheader("🔧 AI Configuration")
        if ollama_client.check_connection():
            st.success("✅ Ollama Connected")
            
            # Model selection
            available_models = ollama_client.get_available_models()
            selected_model = st.selectbox(
                "Select AI Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0
            )
            st.session_state.selected_model = selected_model
            st.info(f"Using: **{selected_model}**")
            
        else:
            st.error("❌ Ollama Not Connected")
            st.markdown("""
            **Troubleshooting:**
            - Run `ollama serve` in terminal
            - Check `ollama list` to see models
            - Test with `ollama run llama2 "Hello"`
            """)
        
        st.divider()
        
        # Chat management
        st.subheader("💬 Chat History")
        
        if st.button("🆕 New Chat", use_container_width=True, type="primary"):
            create_new_chat_session()
            st.rerun()
        
        st.divider()
        
        # Display chat sessions
        for session_id, session_data in list(st.session_state.chat_sessions.items())[::-1]:
            is_active = session_id == st.session_state.current_session
            
            if st.button(
                f"💬 {session_data['title']}",
                key=session_id,
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.current_session = session_id
                st.session_state.processing = False
                st.session_state.uploaded_image = None
                st.session_state.show_ocr_preview = False
                st.rerun()
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        if st.button("🔄 Check Ollama Connection"):
            if ollama_client.check_connection():
                st.success("✅ Ollama is running!")
            else:
                st.error("❌ Cannot connect to Ollama")
            time.sleep(2)
            st.rerun()

    # Main chat area
    st.title("🤖 AI Chat Assistant")
    
    # Display connection status in main area
    if ollama_client.check_connection():
        st.success(f"🚀 Connected to Ollama | Using: **{st.session_state.selected_model}**")
    else:
        st.warning("⚠️ Ollama connection issue - OCR will work but AI responses may be limited")
    
    current_session = st.session_state.current_session
    current_messages = st.session_state.chat_sessions[current_session]["messages"]
    
    # Display messages
    for message in current_messages:
        if message["role"] == "user":
            # Check if message contains image content
            if message.get("has_image"):
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin: 15px 0;'>
                    <div class="user-message">
                        <strong>You</strong><br>
                        <div class="image-preview">
                            📷 <strong>Image Upload</strong><br>
                            <em>Question about uploaded image</em>
                        </div>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='display: flex; justify-content: flex-end; margin: 15px 0;'>
                    <div class="user-message">
                        <strong>You</strong><br>
                        {message["content"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            confidence_badge = display_confidence_badge(
                message.get("confidence", "medium"), 
                message.get("source", "knowledge_base")
            )
            st.markdown(f"""
            <div style='display: flex; justify-content: flex-start; margin: 15px 0;'>
                <div class="ai-message">
                    <strong>AI Assistant</strong> {confidence_badge}<br>
                    {message["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Show OCR preview if we have an uploaded image
    if st.session_state.uploaded_image and st.session_state.show_ocr_preview:
        st.markdown("---")
        st.subheader("📷 Image Uploaded")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(st.session_state.uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            if st.session_state.ocr_extracted_text:
                st.success("✅ Text extracted successfully!")
                
                # User can add a question about the extracted text
                image_question = st.text_input(
                    "Ask about this image:",
                    placeholder="e.g., 'What does this text mean?' or 'Summarize this document'",
                    key="image_question"
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if st.button("💬 Send to Chat", type="primary", use_container_width=True):
                        current_session = st.session_state.current_session
                        current_messages = st.session_state.chat_sessions[current_session]["messages"]
                        
                        # Create the message content - ONLY show the question, not the extracted text
                        if image_question.strip():
                            message_content = image_question
                        else:
                            message_content = "Please analyze the text in this image"
                        
                        current_messages.append({
                            "role": "user", 
                            "content": message_content,
                            "has_image": True,
                            "timestamp": datetime.now().isoformat(),
                            "id": str(uuid.uuid4()),
                            "extracted_text": st.session_state.ocr_extracted_text  # Store internally but don't display
                        })
                        
                        # Update chat title if first message
                        if len(current_messages) == 1:
                            title = "Image Analysis" if not image_question.strip() else image_question[:30] + "..."
                            st.session_state.chat_sessions[current_session]["title"] = title
                        
                        # Reset image state
                        st.session_state.uploaded_image = None
                        st.session_state.show_ocr_preview = False
                        
                        # Set processing state
                        st.session_state.processing = True
                        st.rerun()
                
                with col2:
                    if st.button("❌ Cancel", use_container_width=True):
                        st.session_state.uploaded_image = None
                        st.session_state.show_ocr_preview = False
                        st.rerun()
    
    # Show typing indicator only if processing
    if st.session_state.processing:
        st.markdown("""
        <div style='display: flex; justify-content: flex-start; margin: 15px 0;'>
            <div class="ai-message">
                <strong>AI Assistant</strong><br>
                Thinking
                <span class="typing-indicator">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Process the message after showing the typing indicator
        if current_messages and current_messages[-1]["role"] == "user":
            user_message = current_messages[-1]["content"]
            
            # Check if this is an image-based question and get the extracted text
            extracted_text = current_messages[-1].get("extracted_text", "")
            
            # Build the actual prompt for the AI
            if extracted_text:
                # For image questions, include the extracted text in the prompt but don't show it to user
                actual_prompt = f"{user_message}\n\nText from image:\n{extracted_text}"
            else:
                actual_prompt = user_message
            
            # Simulate processing time
            time.sleep(1)
            
            # Get AI response (Ollama with fallback)
            ai_response, confidence, source = get_ai_response(
                actual_prompt, 
                context_history=current_messages[:-1]  # Exclude current user message
            )
            
            # Add AI response to messages
            current_messages.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now().isoformat(),
                "id": str(uuid.uuid4()),
                "confidence": confidence,
                "source": source
            })
            
            # Reset processing state
            st.session_state.processing = False
            st.rerun()
    
    # Chat input with integrated OCR feature
    st.divider()
    
    # Create a container for the chat input
    chat_container = st.container()
    
    with chat_container:
        # File uploader integrated with chat input
        if OCR_AVAILABLE:
            col1, col2, col3 = st.columns([6, 1, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask me anything...",
                    placeholder="Type your question here or upload an image...",
                    key="chat_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                # File uploader as a button-like element
                uploaded_file = st.file_uploader(
                    "📷",
                    type=['png', 'jpg', 'jpeg'],
                    key="chat_uploader",
                    label_visibility="collapsed",
                    help="Upload an image for OCR"
                )
                
                # Process uploaded file immediately
                if uploaded_file is not None and not st.session_state.uploaded_image:
                    with st.spinner("🔍 Extracting text..."):
                        success, text, confidence, info = process_uploaded_image(uploaded_file)
                        if success:
                            st.session_state.show_ocr_preview = True
                            st.rerun()
                        else:
                            st.error(f"❌ {text}")
            
            with col3:
                send_button = st.button("Send", type="primary", use_container_width=True)
        else:
            col1, col2 = st.columns([6, 1])
            
            with col1:
                user_input = st.text_input(
                    "Ask me anything...",
                    placeholder="Type your question here...",
                    key="chat_input",
                    label_visibility="collapsed"
                )
            
            with col2:
                send_button = st.button("Send", type="primary", use_container_width=True)
    
    # Handle message sending (text only)
    if send_button and user_input.strip() and not st.session_state.uploaded_image:
        # Add user message
        current_messages.append({
            "role": "user", 
            "content": user_input,
            "timestamp": datetime.now().isoformat(),
            "id": str(uuid.uuid4())
        })
        
        # Update chat title if first message
        if len(current_messages) == 1:
            title = user_input[:30] + "..." if len(user_input) > 30 else user_input
            st.session_state.chat_sessions[current_session]["title"] = title
        
        # Set processing state to show typing indicator
        st.session_state.processing = True
        st.rerun()

if __name__ == "__main__":
    main()
