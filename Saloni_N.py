import streamlit as st
import ollama
from datetime import datetime
from PIL import Image
import pytesseract
import io
import base64

# --- Page Configuration ---
st.set_page_config(page_title="AI Chatbot", page_icon="💬", layout="wide")

# --- Custom CSS for Fixed Bottom Input & Styling ---
st.markdown("""
<style>
/* --- General Page Padding --- */
.block-container {
    padding-bottom: 130px !important;
}

/* --- Fixed Input Bar (Bottom) --- */
.chat-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: #f7f7f8;
    border-top: 1px solid #e0e0e0;
    padding: 12px 20px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 10px;
    z-index: 999;
}

/* --- Chat Input Box --- */
textarea[data-testid="stChatInput"] {
    flex: 1;
    border-radius: 25px !important;
    border: 1px solid #ccc;
    padding: 12px 18px;
    font-size: 16px;
    outline: none;
}

/* --- File Upload Button (Circular Icon) --- */
[data-testid="stFileUploader"] section { padding: 0 !important; }
[data-testid="stFileUploader"] div[data-testid="stFileUploaderDropzone"] { border: none !important; padding: 0 !important; }
[data-testid="stFileUploaderDropzone"] label {
    background-color: #ffffff !important;
    border: 1px solid #ccc !important;
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    font-size: 20px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease;
}
[data-testid="stFileUploaderDropzone"] label:hover { background-color: #e9e9e9 !important; }
[data-testid="stFileUploaderDropzone"] div { display: none !important; }

/* --- Chat History Buttons --- */
.sidebar button {
    text-align: left !important;
    border-radius: 10px !important;
    padding: 10px !important;
    background-color: #ffffff !important;
    border: 1px solid #e0e0e0 !important;
    transition: all 0.2s ease;
}
.sidebar button:hover {
    background-color: #f0f0f0 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "chats" not in st.session_state:
    st.session_state.chats = []
if "current_chat" not in st.session_state:
    st.session_state.current_chat = []
if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None

# --- Sidebar Section ---
st.sidebar.title("⚙ Options")

# --- New Chat Button ---
if st.sidebar.button("🆕 New Chat"):
    if st.session_state.current_chat:
        st.session_state.chats.append({
            "id": len(st.session_state.chats) + 1,
            "messages": st.session_state.current_chat.copy(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    st.session_state.current_chat = []
    st.session_state.uploaded_text = ""
    st.session_state.uploaded_image = None

# --- Clear All Chats ---
if st.sidebar.button("🧹 Clear All Chats"):
    st.session_state.chats = []
    st.session_state.current_chat = []
    st.session_state.uploaded_text = ""
    st.session_state.uploaded_image = None

# --- Chat History (Text Only Like ChatGPT) ---
if st.session_state.chats:
    st.sidebar.subheader("📜 Chat History")

    for chat in reversed(st.session_state.chats):
        with st.sidebar.container():
            if chat["messages"]:
                first_msg = next((m for r, m in chat["messages"] if r == "user"), "No message")
                short_preview = (first_msg[:25] + "...") if len(first_msg) > 25 else first_msg
            else:
                short_preview = "Empty chat"

            label = f"💬 {short_preview}\n*{chat['timestamp'].split()[0]}*"
            if st.sidebar.button(label, key=f"chat_{chat['id']}"):
                st.session_state.current_chat = chat["messages"].copy()
                st.session_state.uploaded_image = None

# --- Title ---
st.markdown("<h1 style='text-align:center; color:#333;'>💬 AI Chatbot</h1>", unsafe_allow_html=True)

# --- Display Chat Messages ---
for role, msg in st.session_state.current_chat:
    st.chat_message(role).markdown(msg)

# --- Fixed Bottom Input + File Upload ---
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
col_input, col_upload = st.columns([0.92, 0.08])

with col_input:
    prompt = st.chat_input("Type your message...")

with col_upload:
    uploaded_file = st.file_uploader(
        "📎",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
        key="file_uploader"
    )

st.markdown("</div>", unsafe_allow_html=True)

# --- OCR Processing ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)

        # Convert image to base64
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
        st.session_state.uploaded_image = f"data:image/png;base64,{img_base64}"

        # Extract text using OCR
        text = pytesseract.image_to_string(image)
        st.session_state.uploaded_text = text.strip()
        st.success("✅ Text extracted from image! It will be sent with your next message.")

        # Show preview
        st.image(image, caption="Uploaded Image Preview", width=400)
        st.info(st.session_state.uploaded_text)
    except Exception as e:
        st.error(f"⚠ Could not process image: {e}")

# --- Handle Chat Submission ---
if prompt:
    user_input = prompt
    if st.session_state.uploaded_text:
        user_input += f"\n\n🖼 Extracted from Image:\n{st.session_state.uploaded_text}"
        st.session_state.uploaded_text = ""

    st.chat_message("user").markdown(user_input)
    st.session_state.current_chat.append(("user", user_input))

    try:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ollama.chat(model="phi3:mini", messages=[{"role": "user", "content": prompt}])

                answer = response["message"]["content"]
                st.write(answer)
                st.session_state.current_chat.append(("assistant", answer))
    except Exception as e:
        err = f"⚠ Error: {e}"
        st.chat_message("assistant").markdown(err)
        st.session_state.current_chat.append(("assistant", err))
