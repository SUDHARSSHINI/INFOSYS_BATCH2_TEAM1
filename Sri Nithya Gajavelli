import streamlit as st
import requests
import json
from PIL import Image
import pytesseract
import base64
import io

# ---------------- Tesseract Path ----------------
try:
    # Use a raw string for Windows path and handle potential absence
    pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
except Exception:
    pass

# ---------------- Streamlit Page Config ----------------
st.set_page_config(layout="wide")

# ---------------- Session State ----------------
# NOTE: Chat format now uses ('user_model', clean_prompt) for history and 
# ('user_display', html_content) for UI rendering.
st.session_state.setdefault("chats", {})
st.session_state.setdefault("current_chat", None)
st.session_state.setdefault("chat_counter", 0)
st.session_state.setdefault("extracted_text", "")
st.session_state.setdefault("titles", {})
st.session_state.setdefault("input_widget_counter", 0)
st.session_state.setdefault("input_widget_key", f"input_box_{st.session_state.input_widget_counter}")
st.session_state.setdefault("uploaded_image", None)
st.session_state.setdefault("uploader_key", "file_uploader_0")
st.session_state.setdefault("chat_images", {})
st.session_state.setdefault("ocr_done", False)

# ---------------- Helpers ----------------
def pil_image_to_base64(img, fmt="PNG", max_width=260):
    w, h = img.size
    if w > max_width:
        new_h = int(max_width * h / w)
        img = img.resize((max_width, new_h))
    buffered = io.BytesIO()
    img.save(buffered, format=fmt)
    b64 = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/{fmt.lower()};base64,{b64}"

def uploadedfile_to_pil(uploaded_file):
    uploaded_file.seek(0) 
    return Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

# ---------------- Ollama Backend (Fixed Prompt History) ----------------
def bot_reply(user_msg):
    try:
        history = ""
        if st.session_state.current_chat:
            for sender, text in st.session_state.chats[st.session_state.current_chat]:
                # Only build history from model-specific entries
                if sender == "user_model": 
                    history += f"User: {text}\n"
                elif sender == "bot":
                    # Clean the previous assistant response by removing context headers
                    if "--- Assistant Response ---" in text:
                        core_response = text.split("--- Assistant Response ---")[-1].strip()
                        history += f"Assistant: {core_response}\n"
                    else:
                        history += f"Assistant: {text}\n"

        # The current turn's clean message is already passed as user_msg
        prompt = history + f"User: {user_msg}\nAssistant:"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "llama2", "prompt": prompt},
            stream=True,
        )

        full_reply = ""
        for line in response.iter_lines():
            if line:
                data = json.loads(line.decode("utf-8"))
                if "response" in data:
                    full_reply += data["response"]
                if data.get("done", False):
                    break
        return full_reply.strip() if full_reply else "Sorry, I couldn’t generate a reply."
    except requests.exceptions.ConnectionError:
        return "❌ **Connection Error:** Could not connect to the Ollama backend at `http://localhost:11434`."
    except Exception as e:
        return f"❌ Backend error: {e}"

# ---------------- Sidebar ----------------
st.sidebar.title("💭 Conversations")

if st.sidebar.button("➕ Start New Chat"):
    st.session_state.chat_counter += 1
    new_key = f"chat_{st.session_state.chat_counter}"
    st.session_state.chats[new_key] = []
    st.session_state.current_chat = new_key
    st.session_state.titles[new_key] = f"Chat {st.session_state.chat_counter}"
    st.session_state.chat_images[new_key] = []
    st.session_state.extracted_text = ""
    st.session_state.uploaded_image = None
    st.session_state.ocr_done = False
    st.session_state.uploader_key = f"file_uploader_{st.session_state.chat_counter}_new"
    st.rerun()

if st.session_state.chats:
    keys = list(st.session_state.chats.keys())
    display_names = [st.session_state.titles.get(k, k) for k in keys]
    try:
        selected_index = keys.index(st.session_state.current_chat) if st.session_state.current_chat in keys else 0
    except Exception:
        selected_index = 0
    selected = st.sidebar.radio("Select Chat", display_names, index=selected_index)
    
    for k in keys:
        if st.session_state.titles.get(k, k) == selected:
            if st.session_state.current_chat != k: 
                st.session_state.current_chat = k
                st.session_state.extracted_text = ""
                st.session_state.uploaded_image = None
                st.session_state.ocr_done = False
                st.session_state.uploader_key = f"file_uploader_switch_{st.session_state.current_chat}"
                st.rerun() 
            break

if st.session_state.current_chat and st.session_state.current_chat not in st.session_state.chat_images:
    st.session_state.chat_images[st.session_state.current_chat] = []

# ---------------- Main Area Centering Fix ----------------
col_left_spacer, col_center_content, col_right_spacer = st.columns([1, 4, 1])

with col_center_content:
    # --- Centered Header ---
    st.markdown("<h1 style='text-align: center;'>CODE GEN AI EXPLAINER AND CODE GENERATOR</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center;'>How can I help you?</h3>", unsafe_allow_html=True)

    # --- CSS: Styling and Vertical Alignment Overrides ---
    st.markdown(
        """
        <style>
        /* Center the entire Streamlit main block content to fix left-alignment */
        .main > div {
            max-width: 1200px; 
            padding-left: 20px;
            padding-right: 20px;
        }
        
        /* IMPORTANT: Target the main input row container and force vertical alignment */
        [data-testid="stVerticalBlock"] > div:last-child > div:nth-child(4) > div:nth-child(1) {
             display: flex;
             align-items: center; 
             gap: 0px; 
        }

        /* Adjusted padding for the '+' button's container (col1) */
        [data-testid="stVerticalBlock"] > div:last-child > div:nth-child(4) > div:nth-child(1) > div:nth-child(1) {
             padding-top: 4px; 
             margin-top: 0px !important; 
        }
        
        /* Hide filename and helper text for uploader */
        [data-testid="stFileUploader"] small, 
        [data-testid="stFileUploader"] span, 
        [data-testid="stFileUploader"] > div > label { display: none !important; }

        /* Style and position the uploader button (the '+') */
        [data-testid="stFileUploader"] div[role="button"],
        [data-testid="stFileUploader"] div[role="button"] > div,
        [data-testid="stFileUploader"] section div div {
            width: 40px !important; 
            height: 40px !important;
            min-width: 40px !important;
            min-height: 40px !important;
            
            border-radius: 50% !important;
            background-color: #444 !important;
            color: white !important;
            font-size: 24px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            padding: 0 !important;
            box-shadow: none !important;
            border: none !important;
        }

        /* Insert visible plus sign */
        [data-testid="stFileUploader"] div[role="button"]::after,
        [data-testid="stFileUploader"] section div div::after {
            content: '+' !important;
            color: white !important;
            font-size: 22px !important;
            line-height: 1 !important;
        }
        
        /* Ensure the entire file uploader container is vertically aligned */
        [data-testid="stFileUploader"] {
            display: flex; 
            align-items: center; 
            height: 100%;
        }

        /* Fix for Text Area Height and Styling */
        .clean-input .stTextArea {
            margin-bottom: 0px !important;
        }
        /* Target the actual text input box/textarea element */
        .clean-input .stTextArea>div>div>textarea {
            height: 80px; 
            padding: 10px 12px; 
            font-size: 14px;
            resize: none; 
        }

        /* Send Button Styling */
        button[kind="primary"] { 
            outline: none; 
            height: 40px !important; 
            min-height: 40px !important; 
            margin: 0 !important;
            /* Adjust margin for vertical centering relative to the text area */
            margin-top: 20px !important; 
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # --- Input Row Structure ---
    
    # 1. Uploader Column (Defined OUTSIDE the form)
    col1_uploader = st.columns([0.7, 7.3, 1.2])[0] 

    with col1_uploader:
        uploaded_file = st.file_uploader(
            label="Upload", type=["jpg", "jpeg", "png"], key=st.session_state.uploader_key, 
            label_visibility="collapsed"
        )
        

    # 2. Handle File Upload (Immediate Processing)
    if uploaded_file:
        same_file = (
            st.session_state.uploaded_image is not None
            and uploaded_file.name == st.session_state.uploaded_image.name
            and uploaded_file.size == st.session_state.uploaded_image.size
        )
        
        if not same_file or not st.session_state.ocr_done:
            
            if not st.session_state.current_chat:
                st.session_state.chat_counter += 1
                new_key = f"chat_{st.session_state.chat_counter}"
                st.session_state.chats[new_key] = []
                st.session_state.current_chat = new_key
                st.session_state.titles[new_key] = f"Chat {st.session_state.chat_counter}"
                st.session_state.chat_images[new_key] = []

            st.session_state.uploaded_image = uploaded_file
            try:
                uploaded_file.seek(0)
                pil_img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
                
                with st.spinner("Extracting text from image..."):
                    text = pytesseract.image_to_string(pil_img).strip()
                    st.session_state.extracted_text = text
                    
                uri = pil_image_to_base64(pil_img, fmt="PNG", max_width=260)
                st.session_state.chat_images.setdefault(st.session_state.current_chat, []).append(uri)
                st.session_state.ocr_done = True
                
                st.toast("Image uploaded and text extracted (send your instruction).", icon="✅")
                st.rerun() 
                
            except pytesseract.TesseractNotFoundError:
                st.error("Tesseract not found. OCR disabled.")
                st.session_state.ocr_done = True 
            except Exception as e:
                st.error(f"Error during OCR / thumbnail creation: {e}")
                st.session_state.ocr_done = False


    # 3. Define the Form (FIX: Columns now defined INSIDE the form)
    with st.form(key="main_form", clear_on_submit=False):
        
        # Define the remaining columns *INSIDE* the form to correctly scope the widgets
        col2_input_area, col3_send_button = st.columns([7.3, 1.2], vertical_alignment="center")

        # Text Input
        with col2_input_area:
            st.markdown('<div class="clean-input">', unsafe_allow_html=True)
            user_text = st.text_area(
                "", 
                placeholder="Type your message or instruction here... (Press Enter for new line)", 
                key=st.session_state.input_widget_key,
                height=80, 
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Send button
        with col3_send_button:
            submit = st.form_submit_button("Send", type="primary") 


        # --- Handle Form Submission ---
        if submit:
            key = st.session_state.input_widget_key
            typed = st.session_state.get(key, "").strip()
            extracted = st.session_state.extracted_text.strip()

            if not typed and not extracted and not st.session_state.chat_images.get(st.session_state.current_chat):
                st.warning("Please type a message or upload an image before sending.")
            else:
                if not st.session_state.current_chat:
                    st.session_state.chat_counter += 1
                    new_key = f"chat_{st.session_state.chat_counter}"
                    st.session_state.chats[new_key] = []
                    st.session_state.current_chat = new_key
                    st.session_state.titles[new_key] = "New Chat"
                    st.session_state.chat_images[new_key] = []

                # 1. Prepare combined prompt for the model
                combined_prompt = ""
                if extracted:
                    combined_prompt += f"Extracted text from uploaded image:\n{extracted}"
                if typed:
                    if combined_prompt: combined_prompt += "\n\n"
                    combined_prompt += typed
                
                # 2. Build user display bubble
                thumbnail_html = ""
                q = st.session_state.chat_images.get(st.session_state.current_chat, [])
                if q:
                    uri = q.pop(0) 
                    thumbnail_html = (
                        f"<div style='margin-bottom:8px; text-align:center;'>"
                        f"<a href='{uri}' target='_blank' rel='noopener noreferrer'>"
                        f"<img src='{uri}' style='max-width:260px; max-height:220px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,0.12);'/>"
                        f"</a></div>"
                    )
                user_text_display = typed if typed else '<i>[Image upload without text instruction]</i>'
                user_html = f"{thumbnail_html}<div>{user_text_display}</div>"
                
                # 3. Save two entries: one for display, one for the model history
                st.session_state.chats[st.session_state.current_chat].append(("user_display", user_html))
                st.session_state.chats[st.session_state.current_chat].append(("user_model", combined_prompt)) 

                # 4. Call model
                with st.spinner("Assistant is thinking..."):
                    reply = bot_reply(combined_prompt)

                # 5. Assistant message
                assistant_message = ""
                if extracted:
                    assistant_message += f"--- Extracted Text (For Context) ---\n{extracted}\n\n"
                assistant_message += f"--- Assistant Response ---\n{reply}"
                st.session_state.chats[st.session_state.current_chat].append(("bot", assistant_message)) 

                # 6. Set initial chat title
                if len(st.session_state.chats[st.session_state.current_chat]) == 3:
                    title_preview = (typed if typed else (extracted.splitlines()[0] if extracted else "Chat"))
                    st.session_state.titles[st.session_state.current_chat] = title_preview[:50].strip() + ("..." if len(title_preview) > 50 else "")

                # 7. Reset state for next interaction and clear input
                st.session_state.extracted_text = ""
                st.session_state.uploaded_image = None
                st.session_state.ocr_done = False
                
                st.session_state.input_widget_counter += 1
                st.session_state.input_widget_key = f"input_box_{st.session_state.input_widget_counter}"
                
                st.session_state.uploader_key = f"file_uploader_{st.session_state.chat_counter}_reset_{st.session_state.input_widget_counter}"

                st.rerun()

    # --- Chat Display Area --- 
    if st.session_state.current_chat and st.session_state.chats.get(st.session_state.current_chat):
        for sender, message in st.session_state.chats[st.session_state.current_chat]:
            
            # Skip the model-only history entry
            if sender == "user_model": 
                continue 

            align = "flex-end" if sender == "user_display" else "flex-start" 
            bg_color = "#DCF7C5" if sender == "user_display" else "#F3F3F3"
            
            st.markdown(
                f"""
                <div style="display:flex; justify-content:{align}; margin:6px 0;">
                    <div style="background-color:{bg_color}; padding:12px; border-radius:16px; 
                                max-width:70%; word-wrap:break-word; box-shadow:0 1px 3px rgba(0,0,0,0.1); white-space:pre-wrap;">
                        {message}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<h4 style='text-align: center; color:gray;'>Start a new chat or upload an image to begin!</h4>", unsafe_allow_html=True)

# ---- Footer ---- 
with col_center_content:
    st.markdown(
        "<div style='text-align:center; color:gray; font-size:12px; margin-top:30px;'>"
        "✨ Built with ❤️ by Sri Nithya ✨</div>",
        unsafe_allow_html=True,
    )
