import streamlit as st
from datetime import datetime, timedelta
import requests
import json
import pickle
import os
from PIL import Image
import pytesseract
import io
import base64
import platform
import subprocess

# Set page configuration
st.set_page_config(
    page_title="AI Code Generator - Code Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="auto"
)

# Constants
MAX_CHAT_HISTORY = 15
DATA_FILE = "chat_history.pkl"
IMAGE_DISPLAY_WIDTH = 400

# ===== TESSERACT CONFIG =====
def configure_tesseract():
    """Setup OCR configuration based on operating system"""
    try:
        pytesseract.get_tesseract_version()
        return True
    except:
        pass
    
    # Try to configure for Windows
    if platform.system() == "Windows":
        paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        ]
        for path in paths:
            if os.path.exists(path):
                try:
                    pytesseract.pytesseract.tesseract_cmd = path
                    # Test the configuration
                    subprocess.run([path, '--version'], timeout=10, capture_output=True)
                    return True
                except:
                    continue
    return False

def optimize_image_for_ocr(image):
    """Optimize image for better OCR results"""
    try:
        # Convert to grayscale for better OCR
        if image.mode != "L":
            image = image.convert("L")
        # Resize if too large
        if image.width > 1200:
            ratio = 1200 / image.width
            image = image.resize((1200, int(image.height * ratio)), Image.Resampling.LANCZOS)
    except Exception as e:
        st.error(f"Image optimization error: {e}")
    return image

def resize_image_for_display(image, target_width=IMAGE_DISPLAY_WIDTH):
    """Resize image consistently for display purposes"""
    try:
        if image.width > target_width:
            ratio = target_width / image.width
            new_height = int(image.height * ratio)
            image = image.resize((target_width, new_height), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        st.error(f"Image resize error: {e}")
        return image

def extract_text_from_image(image):
    """Extract text from uploaded image using OCR with optimization"""
    try:
        image = optimize_image_for_ocr(image)
        text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
        return text.strip()
    except Exception as e:
        return f"Error extracting text: {str(e)}"

def clear_ocr_data():
    """Clear OCR-related session state data"""
    keys_to_remove = [
        'ocr_extracted_text', 
        'ocr_image_uploaded', 
        'ocr_context_ready',
        'ocr_active_image',
        'user_text_input',
        'current_uploaded_image'
    ]
    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

def enhance_system_prompt_with_ocr():
    """Enhance the system prompt to handle OCR content"""
    ocr_context = ""
    if st.session_state.get('ocr_extracted_text'):
        ocr_context = f"""
        
USER UPLOADED IMAGE CONTEXT:
The user has uploaded an image with the following extracted text:
"{st.session_state.ocr_extracted_text}"

Please analyze this content and answer the user's question based on the extracted text from the image.
"""
    return ocr_context

def save_chats_to_file():
    """Save chat history to pickle file"""
    try:
        data = {
            "chat_history": st.session_state.chat_history,
            "next_chat_id": st.session_state.next_chat_id
        }
        with open(DATA_FILE, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        st.error(f"Error saving to file: {e}")
        return False

def load_chats_from_file():
    """Load chat history from pickle file"""
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'rb') as f:
                data = pickle.load(f)
            return data.get("chat_history", []), data.get("next_chat_id", 1)
        return [], 1
    except Exception as e:
        st.error(f"Error loading from file: {e}")
        return [], 1

def format_response_with_code(text):
    """Format response with proper code block styling"""
    parts = text.split('```')
    formatted_text = ""
    
    for i, part in enumerate(parts):
        if i % 2 == 0:
            formatted_text += part
        else:
            lines = part.split('\n', 1)
            if len(lines) > 1 and lines[0].strip():
                language = lines[0].strip()
                code_content = lines[1]
            else:
                language = 'text'
                code_content = part
            
            formatted_text += f"\n```{language}\n{code_content}\n```\n"
    
    return formatted_text

def organize_chats_by_date(chats):
    """Organize chats into date-based sections"""
    today = datetime.now().date()
    yesterday = today - timedelta(days=1)
    last_week = today - timedelta(days=7)
    
    sections = {
        "Today": [], "Yesterday": [], "Previous 7 Days": [], "Older": []
    }
    
    for chat in chats:
        chat_date_str = chat.get("last_updated", chat["date"])
        try:
            chat_date = datetime.strptime(chat_date_str, "%Y-%m-%d %H:%M").date()
        except:
            chat_date = datetime.strptime(chat_date_str.split()[0], "%Y-%m-%d").date()
        
        if chat_date == today:
            sections["Today"].append(chat)
        elif chat_date == yesterday:
            sections["Yesterday"].append(chat)
        elif chat_date >= last_week:
            sections["Previous 7 Days"].append(chat)
        else:
            sections["Older"].append(chat)
    
    return sections

def initialize_session_state():
    """Initialize all session state variables with persistent data"""
    # Load from file storage
    chat_history, next_chat_id = load_chats_from_file()
    
    default_state = {
        "messages": [],
        "chat_history": chat_history,
        "current_chat_id": None,
        "chat_started": False,
        "next_chat_id": next_chat_id,
        "search_query": "",
        "ollama_url": "http://localhost:11434",
        "model": "llama2",
        "temperature": 0.3,
        "max_tokens": 2000,
        "use_streaming": True,
        "max_chat_history": MAX_CHAT_HISTORY,
        "ocr_extracted_text": "",
        "ocr_image_uploaded": False,
        "ocr_context_ready": False,
        "ocr_active_image": None,
        "ollama_connected": False,
        "available_models": [],
        "tesseract_configured": False,
        "user_text_input": "",
        "current_uploaded_image": None,
        "show_uploader": False
    }
    
    for key, value in default_state.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize session state
initialize_session_state()

# Initialize OCR
if not st.session_state.tesseract_configured:
    st.session_state.tesseract_configured = configure_tesseract()

# ===== OLLAMA CONNECTION FUNCTIONS =====
def test_ollama_connection():
    """Test connection to Ollama and get available models"""
    try:
        url = f"{st.session_state.ollama_url}/api/tags"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            st.session_state.available_models = [model['name'] for model in data.get('models', [])]
            st.session_state.ollama_connected = True
            return True
        else:
            st.session_state.ollama_connected = False
            return False
    except:
        st.session_state.ollama_connected = False
        return False

def get_available_models():
    """Get list of available models from Ollama"""
    try:
        url = f"{st.session_state.ollama_url}/api/tags"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model['name'] for model in data.get('models', [])]
        return []
    except:
        return []

def get_ai_response(user_input, conversation_history=None):
    """Generate AI response using Ollama local LLM with OCR context"""
    try:
        messages = []
        
        system_message = """You are an expert AI assistant specialized in code generation and programming. 
When generating code, follow these guidelines:
1. Provide complete, runnable code examples
2. Include proper syntax highlighting markers (e.g., ```python, ```javascript, etc.)

For non-code questions, provide clear, concise, and accurate responses."""
        
        ocr_context = enhance_system_prompt_with_ocr()
        system_message += ocr_context
        
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        url = f"{st.session_state.ollama_url}/api/chat"
        
        # Try multiple models if the default one fails
        models_to_try = [st.session_state.model]
        if st.session_state.available_models:
            # Add other available models to try
            for model in st.session_state.available_models:
                if model not in models_to_try:
                    models_to_try.append(model)
        
        for model in models_to_try[:3]:  # Try up to 3 models
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": st.session_state.temperature,
                    "num_predict": st.session_state.max_tokens
                }
            }
            
            try:
                response = requests.post(url, json=payload, timeout=120)
                
                if response.status_code == 200:
                    response_data = response.json()
                    return response_data['message']['content']
                elif response.status_code == 404:
                    continue  # Try next model if this one doesn't exist
                else:
                    # If this is the last model to try, return error
                    if model == models_to_try[-1]:
                        return f"Error: Ollama API returned status code {response.status_code}. Please check if the model '{model}' is downloaded (run: ollama pull {model.split(':')[0]})"
            except requests.exceptions.ConnectionError:
                return f"Connection error: Cannot connect to Ollama at {st.session_state.ollama_url}. Please make sure Ollama is running."
            except requests.exceptions.Timeout:
                return "Request timeout: Ollama is taking too long to respond."
            except Exception as e:
                if model == models_to_try[-1]:
                    return f"An error occurred: {str(e)}"
        
        return "Error: No working models found. Please make sure you have at least one model downloaded (e.g., ollama pull llama2)."
        
    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_ai_response_streamed(user_input, conversation_history=None):
    """Generate AI response using Ollama with streaming for better UX"""
    try:
        messages = []
        
        system_message = """You are an expert AI assistant specialized in code generation and programming. 
When generating code, follow these guidelines:
1. Provide complete, runnable code examples
2. Include proper syntax highlighting markers (e.g., ```python, ```javascript, etc.)

For non-code questions, provide clear, concise, and accurate responses."""
        
        ocr_context = enhance_system_prompt_with_ocr()
        system_message += ocr_context
        
        messages.append({
            "role": "system",
            "content": system_message
        })
        
        if conversation_history:
            for msg in conversation_history:
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        messages.append({
            "role": "user",
            "content": user_input
        })
        
        url = f"{st.session_state.ollama_url}/api/chat"
        
        # Try multiple models if the default one fails
        models_to_try = [st.session_state.model]
        if st.session_state.available_models:
            for model in st.session_state.available_models:
                if model not in models_to_try:
                    models_to_try.append(model)
        
        for model in models_to_try[:3]:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": st.session_state.temperature,
                    "num_predict": st.session_state.max_tokens
                }
            }
            
            try:
                response = requests.post(url, json=payload, timeout=180, stream=True)
                
                if response.status_code == 200:
                    full_response = ""
                    for line in response.iter_lines():
                        if line:
                            try:
                                line_data = json.loads(line)
                                if 'message' in line_data and 'content' in line_data['message']:
                                    content = line_data['message']['content']
                                    full_response += content
                                    yield content
                                elif 'done' in line_data and line_data['done']:
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response
                elif response.status_code == 404:
                    continue  # Try next model
                else:
                    if model == models_to_try[-1]:
                        yield f"Error: Ollama API returned status code {response.status_code}. Please check if the model '{model}' is downloaded."
            except requests.exceptions.ConnectionError:
                if model == models_to_try[-1]:
                    yield f"Connection error: Cannot connect to Ollama at {st.session_state.ollama_url}"
            except requests.exceptions.Timeout:
                if model == models_to_try[-1]:
                    yield "Request timeout: Ollama is taking too long to respond."
            except Exception as e:
                if model == models_to_try[-1]:
                    yield f"An error occurred: {str(e)}"
        
        yield "Error: No working models found. Please download a model first (e.g., ollama pull llama2)."
        
    except Exception as e:
        yield f"An error occurred: {str(e)}"

def save_persistent_data():
    """Save chat history to persistent storage"""
    try:
        return save_chats_to_file()
    except Exception as e:
        st.error(f"Error saving persistent data: {e}")
        return False

def manage_chat_history():
    """Manage chat history to ensure it doesn't exceed the maximum limit"""
    if len(st.session_state.chat_history) > st.session_state.max_chat_history:
        chats_to_remove = len(st.session_state.chat_history) - st.session_state.max_chat_history
        st.session_state.chat_history = st.session_state.chat_history[:-chats_to_remove]
        
        save_persistent_data()
        
        if chats_to_remove > 0:
            st.warning(f"⚠️ Memory is full! Removed {chats_to_remove} oldest chat(s) to make space.")

def create_new_chat_id():
    """Generate a new unique chat ID"""
    chat_id = st.session_state.next_chat_id
    st.session_state.next_chat_id += 1
    return chat_id

def get_or_create_chat_session():
    """Get current chat session or create a new one if none exists"""
    if st.session_state.current_chat_id is None:
        st.session_state.current_chat_id = create_new_chat_id()
        st.session_state.chat_started = True
        return st.session_state.current_chat_id
    return st.session_state.current_chat_id

def update_chat_title(first_user_message):
    """Update chat title based on first user message"""
    chat_title = first_user_message[:30] + "..." if len(first_user_message) > 30 else first_user_message
    if not chat_title:
        chat_title = "New Chat"
    return chat_title

def save_current_chat(update_timestamp=True):
    """Save the current chat to history - only saves when there are actual messages"""
    if st.session_state.messages and st.session_state.chat_started:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
        chat_id = st.session_state.current_chat_id
        
        existing_chat_index = -1
        for i, chat in enumerate(st.session_state.chat_history):
            if chat["id"] == chat_id:
                existing_chat_index = i
                break
        
        first_user_message = ""
        if existing_chat_index == -1:
            for msg in st.session_state.messages:
                if msg["role"] == "user":
                    first_user_message = msg["content"]
                    break
            chat_title = update_chat_title(first_user_message)
            
            image_data = st.session_state.get('ocr_active_image')
            extracted_text = st.session_state.get('ocr_extracted_text', '')
            
            chat_data = {
                "id": chat_id,
                "title": chat_title,
                "messages": st.session_state.messages.copy(),
                "date": current_time,
                "last_updated": current_time,
                "image_data": image_data,
                "extracted_text": extracted_text
            }
            
            st.session_state.chat_history.insert(0, chat_data)
            
        else:
            st.session_state.chat_history[existing_chat_index]["messages"] = st.session_state.messages.copy()
            
            if st.session_state.get('ocr_active_image'):
                st.session_state.chat_history[existing_chat_index]["image_data"] = st.session_state.ocr_active_image
                st.session_state.chat_history[existing_chat_index]["extracted_text"] = st.session_state.get('ocr_extracted_text', '')
            
            if update_timestamp:
                st.session_state.chat_history[existing_chat_index]["last_updated"] = current_time
                updated_chat = st.session_state.chat_history.pop(existing_chat_index)
                st.session_state.chat_history.insert(0, updated_chat)
        
        manage_chat_history()
        save_persistent_data()

def start_new_chat():
    """Start a new chat session - creates a completely separate chat"""
    if st.session_state.messages and st.session_state.chat_started and st.session_state.current_chat_id is not None:
        save_current_chat(update_timestamp=False)
    
    new_chat_id = create_new_chat_id()
    
    st.session_state.messages = []
    st.session_state.current_chat_id = new_chat_id
    st.session_state.chat_started = False
    st.session_state.search_query = ""
    clear_ocr_data()
    st.session_state.user_text_input = ""
    st.session_state.show_uploader = False

def delete_chat(chat_id):
    """Delete a chat from history"""
    st.session_state.chat_history = [chat for chat in st.session_state.chat_history if chat["id"] != chat_id]
    
    if st.session_state.current_chat_id == chat_id:
        st.session_state.messages = []
        st.session_state.current_chat_id = None
        st.session_state.chat_started = False
        clear_ocr_data()
        st.session_state.user_text_input = ""
        st.session_state.show_uploader = False
    
    save_persistent_data()
    
    st.success("Chat deleted successfully!")

def delete_all_chats():
    """Delete all chat history"""
    st.session_state.chat_history = []
    st.session_state.messages = []
    st.session_state.current_chat_id = None
    st.session_state.chat_started = False
    st.session_state.next_chat_id = 1
    clear_ocr_data()
    st.session_state.user_text_input = ""
    st.session_state.show_uploader = False
    
    save_persistent_data()
    
    try:
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
    except:
        pass
    
    st.success("All chat history cleared!")

def load_chat(chat_id):
    """Load a specific chat into the current view"""
    for chat in st.session_state.chat_history:
        if chat["id"] == chat_id:
            st.session_state.current_chat_id = chat_id
            st.session_state.messages = chat["messages"].copy()
            st.session_state.chat_started = True
            
            if chat.get("image_data"):
                st.session_state.ocr_active_image = chat["image_data"]
                st.session_state.ocr_extracted_text = chat.get("extracted_text", "")
                st.session_state.ocr_context_ready = True
            break

def filter_chats(search_query):
    """Filter chats based on search query"""
    if not search_query:
        return st.session_state.chat_history
    
    search_lower = search_query.lower()
    filtered_chats = []
    
    for chat in st.session_state.chat_history:
        if search_lower in chat["title"].lower():
            filtered_chats.append(chat)
        else:
            for message in chat["messages"]:
                if search_lower in message["content"].lower():
                    filtered_chats.append(chat)
                    break
    
    return filtered_chats

def process_uploaded_image(uploaded_image):
    """Process uploaded image and extract text without showing OCR processing"""
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        
        # Extract text silently without showing spinner
        extracted_text = extract_text_from_image(image)
        
        st.session_state.ocr_extracted_text = extracted_text
        st.session_state.ocr_image_uploaded = True
        st.session_state.ocr_context_ready = True
        
        # Resize image for consistent small display before storing
        resized_image = resize_image_for_display(image, IMAGE_DISPLAY_WIDTH)
        img_byte_arr = io.BytesIO()
        resized_image.save(img_byte_arr, format='PNG')
        current_image_data = img_byte_arr.getvalue()
        
        st.session_state.ocr_active_image = current_image_data
        st.session_state.current_uploaded_image = uploaded_image
        
        return True, extracted_text
    return False, ""

# Test connection on startup
if not st.session_state.ollama_connected:
    test_ollama_connection()

# Sidebar
with st.sidebar:
    st.title("🤖 AI Code Generator")
    
    # Connection status
    if st.session_state.ollama_connected:
        if st.session_state.available_models:
            st.caption(f"Models: {len(st.session_state.available_models)} available")
    else:
        st.error("❌ Ollama Not Connected")
        st.markdown("""
        **To fix this:**
        1. Install Ollama from [ollama.ai](https://ollama.ai)
        2. Run: `ollama serve`
        """)
    
    if st.button("+ New Chat", use_container_width=True, type="primary"):
        start_new_chat()
        st.rerun()
    
    if st.session_state.chat_history:
        if st.button("🗑️ Clear All Chats", use_container_width=True, type="secondary"):
            delete_all_chats()
            st.rerun()
    
    search_query = st.text_input(
        "Search chats",
        value=st.session_state.search_query,
        placeholder="Search chat titles...",
        key="search_input"
    )
    
    if search_query != st.session_state.search_query:
        st.session_state.search_query = search_query
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("Chat History")
    
    filtered_chats = filter_chats(st.session_state.search_query)
    
    if filtered_chats:
        chat_sections = organize_chats_by_date(filtered_chats)
        
        for section_name, section_chats in chat_sections.items():
            if section_chats:
                st.markdown(f"**{section_name}**")
                
                for chat in section_chats:
                    is_active = st.session_state.current_chat_id == chat["id"]
                    
                    col1, col2 = st.columns([0.8, 0.2])
                    
                    with col1:
                        button_type = "primary" if is_active else "secondary"
                        
                        chat_date_str = chat.get("last_updated", chat["date"])
                        try:
                            chat_date = datetime.strptime(chat_date_str, "%Y-%m-%d %H:%M")
                            formatted_date = chat_date.strftime("%b %d, %Y at %H:%M")
                        except:
                            formatted_date = chat_date_str
                        
                        chat_title = chat['title']
                        if chat.get("image_data"):
                            chat_title = f"🖼️ {chat_title}"
                        
                        if st.button(
                            f"{chat_title}",
                            key=f"load_{chat['id']}",
                            use_container_width=True,
                            type=button_type,
                            help=f"Last updated: {formatted_date}"
                        ):
                            load_chat(chat["id"])
                            st.rerun()
                    
                    with col2:
                        if st.button(
                            "🗑️",
                            key=f"delete_{chat['id']}",
                            help=f"Delete this chat",
                            use_container_width=True
                        ):
                            delete_chat(chat["id"])
                            st.rerun()
                
                st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    else:
        if st.session_state.search_query:
            st.warning(f"No chats found for '{st.session_state.search_query}'")
        else:
            st.info("No chat history yet. Start a new chat to begin!")

# Main chat area
st.title("🤖 AI Code Generator")

if not st.session_state.ollama_connected:
    st.error(f"⚠️ Cannot connect to Ollama at {st.session_state.ollama_url}")
    st.info("**To get started:**")
    st.code("""
# Install Ollama first from https://ollama.ai
# Then run these commands:
ollama serve
ollama pull llama2
    """)

# Display current chat title
if st.session_state.current_chat_id and st.session_state.messages:
    current_chat = next((chat for chat in st.session_state.chat_history 
                        if chat["id"] == st.session_state.current_chat_id), None)
    if current_chat:
        title_prefix = "🖼️ " if current_chat.get("image_data") else ""
        st.subheader(f"{title_prefix}{current_chat['title']}")

# Create a container for ALL chat messages (both existing and new)
chat_container = st.container()

with chat_container:
    # Display all existing chat messages
    if st.session_state.messages:
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                # User messages on the right side using columns
                col1, col2 = st.columns([2, 2])
                with col2:
                    with st.chat_message("user"):
                        # Check if this message has image data
                        if message.get("image_data"):
                            try:
                                # Display the image with small consistent size
                                image = Image.open(io.BytesIO(message["image_data"]))
                                st.image(image, caption="Uploaded Image", use_container_width=False, width=IMAGE_DISPLAY_WIDTH)
                            except Exception as e:
                                st.write("📷 Image uploaded")
                        
                        # Always display the user's text content
                        if message.get("content"):
                            st.write(message["content"])
            else:
                # Assistant messages on the left side using columns
                col1, col2 = st.columns([4, 1])
                with col1:
                    with st.chat_message("assistant"):
                        content = message["content"]
                        
                        if '```' in content:
                            parts = content.split('```')
                            for i, part in enumerate(parts):
                                if i % 2 == 0:
                                    st.write(part)
                                else:
                                    lines = part.split('\n', 1)
                                    if len(lines) > 1 and lines[0].strip() in ['python', 'javascript', 'java', 'cpp', 'html', 'css', 'sql', 'bash', 'json', 'yaml', 'xml']:
                                        language = lines[0].strip()
                                        code_content = lines[1]
                                    else:
                                        language = 'text'
                                        code_content = part
                                    
                                    st.code(code_content, language=language)
                        else:
                            st.write(content)

# Input section at the bottom with separator
st.markdown("---")

# Combined input area for both text and image
col1, col2 = st.columns([4, 1])

with col1:
    user_input = st.chat_input(
        "Type your message here...",
        key="user_input_chat"
    )

with col2:
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    
    # Add an upload button that triggers the file uploader with negative margin
    if st.button("📷 Upload File", use_container_width=True, help="Upload an image for OCR analysis"):
        # Toggle the uploader visibility
        st.session_state.show_uploader = not st.session_state.get('show_uploader', False)

    # File uploader (only shown when button is clicked)
    if st.session_state.get('show_uploader', False):
        uploaded_image = st.file_uploader(
            "📷 Upload Image",
            type=["jpg", "jpeg", "png"],
            key="image_uploader",
            help="Upload an image to analyze",
            label_visibility="collapsed"
        )
        
        # Show small image preview when image is uploaded
        if uploaded_image is not None:
            try:
                image = Image.open(uploaded_image)
                resized_image = resize_image_for_display(image, IMAGE_DISPLAY_WIDTH)
                # st.image(resized_image, caption="Image Preview", use_container_width=False, width=IMAGE_DISPLAY_WIDTH)
            except Exception as e:
                st.error(f"Error previewing image: {e}")
    
        # Add a button to close the uploader
        # if st.button("Close Upload", use_container_width=True):
        #     st.session_state.show_uploader = False
        #     st.rerun()
    else:
        uploaded_image = None

# Process when user presses Enter in chat input
if user_input:
    if not st.session_state.ollama_connected:
        st.error(f"Cannot connect to Ollama. Please make sure it's running at {st.session_state.ollama_url}")
        st.stop()
    
    chat_id = get_or_create_chat_session()
    
    if (len(st.session_state.chat_history) >= st.session_state.max_chat_history and 
        not any(chat["id"] == chat_id for chat in st.session_state.chat_history)):
        st.error(f"⚠️ Memory is full! Maximum {st.session_state.max_chat_history} chats allowed. Please delete some chats to create new ones.")
        st.stop()
    
    if not st.session_state.chat_started:
        st.session_state.chat_started = True
    
    # Process image if uploaded (silently without showing OCR processing)
    image_data = None
    extracted_text = ""
    if uploaded_image is not None:
        success, extracted_text = process_uploaded_image(uploaded_image)
        if success:
            image_data = st.session_state.ocr_active_image
    
    # Create message content
    message_content = user_input
    
    # Add user message to chat
    if uploaded_image is not None:
        message_data = {
            "role": "user", 
            "content": message_content,
            "image_data": image_data,
            "extracted_text": extracted_text
        }
    else:
        message_data = {
            "role": "user", 
            "content": message_content
        }
    
    st.session_state.messages.append(message_data)
    
    # Save chat immediately after adding user message
    save_current_chat(update_timestamp=True)
    
    # Display user message immediately on the RIGHT side using columns
    col1, col2 = st.columns([2, 2])
    with col2:
        with st.chat_message("user"):
            if uploaded_image is not None:
                try:
                    image = Image.open(io.BytesIO(image_data))
                    st.image(image, caption="Uploaded File", use_container_width=False, width=IMAGE_DISPLAY_WIDTH)
                except Exception as e:
                    st.write("📷 Image uploaded")
            
            if message_content:
                st.write(message_content)

    # Generate and display AI response immediately below user input on the LEFT side using columns
    col1, col2 = st.columns([4, 1])
    with col1:
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                # Combine image context with user input
                full_prompt = message_content
                if extracted_text and not full_prompt.strip():
                    full_prompt = "What can you tell me about the content of this image?"
                elif extracted_text and full_prompt.strip():
                    full_prompt = full_prompt
                
                if st.session_state.use_streaming:
                    message_placeholder = st.empty()
                    full_response = ""
                    
                    try:
                        for chunk in get_ai_response_streamed(full_prompt, st.session_state.messages[:-1]):
                            full_response += chunk
                            if '```' in full_response:
                                message_placeholder.markdown(format_response_with_code(full_response + "▌"), unsafe_allow_html=True)
                            else:
                                message_placeholder.write(full_response + "▌")
                        
                        if '```' in full_response:
                            message_placeholder.markdown(format_response_with_code(full_response), unsafe_allow_html=True)
                        else:
                            message_placeholder.write(full_response)
                        ai_response = full_response
                        
                    except Exception as e:
                        error_msg = f"Error during streaming: {str(e)}"
                        st.error(error_msg)
                        ai_response = error_msg
                else:
                    ai_response = get_ai_response(full_prompt, st.session_state.messages[:-1])
                    if '```' in ai_response:
                        st.markdown(format_response_with_code(ai_response), unsafe_allow_html=True)
                    else:
                        st.write(ai_response)

    # Add AI response to messages
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
    
    # Save chat with AI response
    save_current_chat(update_timestamp=True)
    
    # Clear the uploaded image after sending and hide uploader
    st.session_state.current_uploaded_image = None
    st.session_state.show_uploader = False
    
    # Rerun to update the conversation history
    st.rerun()

# Memory warning
if len(st.session_state.chat_history) >= st.session_state.max_chat_history:
    st.warning(f"""
    ⚠️ **Memory Full!** 
    
    You've reached the maximum limit of {st.session_state.max_chat_history} chats. 
    Delete some old chats to create new ones.
    """)