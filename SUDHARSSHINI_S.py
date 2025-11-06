import streamlit as st
import uuid
from datetime import datetime
import ollama
from PIL import Image
import pytesseract
import base64
from io import BytesIO

# ----------------- CONFIG -----------------
st.set_page_config(page_title="ChatGPT Clone with Simple OCR", page_icon="🤖", layout="wide")

# ----------------- OCR CONFIG -----------------
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ----------------- STATE -----------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.chats[cid] = {"title": "New Chat", "messages": [], "last_updated": datetime.now()}
if "search_term" not in st.session_state:
    st.session_state.search_term = ""
if "rename_mode" not in st.session_state:
    st.session_state.rename_mode = None
if "show_uploader" not in st.session_state:
    st.session_state.show_uploader = False
if "ocr_text" not in st.session_state:
    st.session_state.ocr_text = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# ----------------- HELPERS -----------------
def create_chat():
    cid = str(uuid.uuid4())
    st.session_state.current_chat = cid
    st.session_state.chats[cid] = {"title": "New Chat", "messages": [], "last_updated": datetime.now()}

def image_to_base64(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def add_message(role, text, image_data=None):
    chat = st.session_state.chats[st.session_state.current_chat]
    msg = {
        "role": role,
        "text": text,
        "time": datetime.now().strftime("%H:%M")
    }
    if image_data:
        msg["image"] = image_data
    chat["messages"].append(msg)
    chat["last_updated"] = datetime.now()
    if chat["title"] == "New Chat" and role == "user":
        chat["title"] = text[:25] + ("..." if len(text) > 25 else "")

# ✅ Simple OCR Extractor
def extract_text_simple(image):
    try:
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f" OCR Error: {e}"

# ✅ Model Response
def bot_reply(prompt, ocr_context=None):
    try:
        if ocr_context:
            full_prompt = f"""You are an AI assistant analyzing text extracted from an image.

Extracted OCR text:
\"\"\"{ocr_context}\"\"\"

Now, answer the user's question based on this text:
{prompt}
"""
        else:
            full_prompt = prompt

        with st.spinner(" Thinking..."):
            response = ollama.chat(
                model="llama3.2:1b",
                messages=[{"role": "user", "content": full_prompt}],
                stream=False
            )
        msg = response.get("message", {}).get("content", "")
        return msg.strip() if msg else "⚠️ No response generated."
    except Exception as e:
        return f"⚠️ Error generating response: {e}"

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("## ChatGPT Clone with OCR")
    if st.button(" New chat"):
        create_chat()

    st.text_input(" Search chats", key="search_term", placeholder="Search chat titles")

    st.markdown("###  Chats")
    for cid, chat in sorted(st.session_state.chats.items(), key=lambda x: x[1]["last_updated"], reverse=True):
        if st.session_state.search_term.lower() in chat["title"].lower():
            col1, col2 = st.columns([7, 1])
            with col1:
                if st.button(chat["title"], key=f"chat_{cid}"):
                    st.session_state.current_chat = cid
            with col2:
                with st.popover("⋮"):
                    if st.button("Rename", key=f"edit_{cid}"):
                        st.session_state.rename_mode = cid
                        st.rerun()
                    if st.button("Delete", key=f"delete_{cid}"):
                        del st.session_state.chats[cid]
                        st.rerun()

            if st.session_state.rename_mode == cid:
                new_title = st.text_input("Rename chat", value=chat["title"], key=f"rename_input_{cid}")
                if st.button(" Save", key=f"save_{cid}"):
                    st.session_state.chats[cid]["title"] = new_title.strip() or chat["title"]
                    st.session_state.rename_mode = None
                    st.rerun()

# ----------------- MAIN CHAT DISPLAY -----------------
current = st.session_state.chats[st.session_state.current_chat]

if not current["messages"]:
    st.markdown(
        "<div style='text-align:center; margin-top:30vh;'>"
        "<h2>What can I help you with today?</h2>"
        "</div>",
        unsafe_allow_html=True,
    )
else:
    for msg in current["messages"]:
        align = "right" if msg["role"] == "user" else "left"

        # ✅ Pale blue container for AI responses
        if msg["role"] == "bot":
            bg_color = "#DDEEFF"  # Pale blue shade
            container_style = (
                f"border-radius:16px; padding:12px 16px; margin:10px 0; "
                f"background-color:{bg_color}; box-shadow:0 0 6px rgba(0,0,0,0.03); "
                f"max-width:80%; text-align:left;"
            )
        else:
            container_style = "background:#DCF8C6; padding:8px 12px; border-radius:12px; display:inline-block;"

        html = f"<div style='text-align:{align}; margin:8px;'>"
        if "image" in msg:
            html += f"<img src='data:image/png;base64,{msg['image']}' width='220' style='border-radius:12px; margin-bottom:8px;'/><br>"
        html += f"<div style='{container_style}'>{msg['text']}</div></div>"
        st.markdown(html, unsafe_allow_html=True)

# ----------------- FLOATING INPUT -----------------
st.markdown("""
<style>
.stChatFloatingInput {
    position: fixed;
    bottom: 0;
    left: 25%;
    right: 25%;
    background: white;
    padding: 10px 16px;
    border-top: 1px solid #ddd;
    border-radius: 16px;
    box-shadow: 0 0 12px rgba(0,0,0,0.08);
}
</style>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="stChatFloatingInput">', unsafe_allow_html=True)
    cols = st.columns([1, 8, 1])
    with cols[0]:
        if st.button("➕", key="open_ocr", help="Upload image for OCR", use_container_width=True):
            st.session_state.show_uploader = not st.session_state.show_uploader
    with cols[1]:
        prompt = st.chat_input("Ask anything (you can refer to uploaded image)...")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- IMAGE UPLOAD & OCR (Simple Mode) -----------------
if st.session_state.show_uploader:
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.session_state.uploaded_image = image
        st.image(image, width=300, caption="🖼️ Uploaded Image")

        # ✅ Simple OCR extraction only
        extracted_text = extract_text_simple(image)
        st.session_state.ocr_text = extracted_text

        if extracted_text:
            st.success("✅ Text extracted successfully!")
            st.text_area(" Extracted Text", extracted_text, height=200)
        else:
            st.warning("⚠️ No readable text found. Try clearer or higher-contrast images.")

# ----------------- USER INPUT HANDLING -----------------
if prompt:
    # ✅ Only use OCR text if an image is uploaded at this moment
    if st.session_state.uploaded_image and st.session_state.ocr_text.strip():
        ocr_context = st.session_state.ocr_text
        image_data = image_to_base64(st.session_state.uploaded_image)
    else:
        ocr_context = None
        image_data = None

    add_message("user", prompt, image_data=image_data)

    # ✅ Show OCR text only if image is uploaded now
    if ocr_context:
        add_message("bot", f" Extracted OCR Text:\n\n{ocr_context}")

    # ✅ Get AI reply — based on OCR if image is uploaded, else Ollama alone
    reply = bot_reply(prompt, ocr_context)
    add_message("bot", f" AI Response:\n\n{reply}")

    # ✅ Reset uploader state after use — so future questions don’t reuse old OCR
    st.session_state.show_uploader = False
    st.session_state.uploaded_image = None
    st.session_state.ocr_text = ""

    st.rerun()
