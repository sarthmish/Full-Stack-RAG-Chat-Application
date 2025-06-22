import gradio as gr
import os
import time
from datetime import datetime
from ui_backend import UIBackend

# Initialize backend
backend = UIBackend(directory_name='data_store')

# Enhanced session state with conversation management
class ChatManager:
    def __init__(self):
        self.conversations = {}
        self.current_chat_id = None
        self.chat_counter = 1
        self.create_new_chat()  # Start with an initial chat

    def create_new_chat(self, title=None):
        chat_id = f"chat_{self.chat_counter}"
        if not title:
            title = f"New Chat {self.chat_counter}"

        self.conversations[chat_id] = {
            "title": title,
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "rag_active": False,
            "selected_docs": []
        }
        self.current_chat_id = chat_id
        self.chat_counter += 1
        return chat_id

    def switch_to_chat(self, chat_id):
        if chat_id in self.conversations:
            self.current_chat_id = chat_id
            return True
        return False

    def get_current_chat(self):
        if not self.current_chat_id or self.current_chat_id not in self.conversations:
            # Fallback to the most recent chat or create a new one
            if self.conversations:
                self.current_chat_id = list(self.conversations.keys())[0]
            else:
                self.create_new_chat()
        return self.conversations[self.current_chat_id]

    def add_message(self, role, content):
        chat = self.get_current_chat()
        chat["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def update_chat_title(self, title):
        chat = self.get_current_chat()
        chat["title"] = title

    def get_chat_history_for_display(self):
        chat = self.get_current_chat()
        history = []
        for msg in chat["messages"]:
            if msg["role"] == "user":
                history.append([msg["content"], None])
            elif msg["role"] == "system":
                history.append([None, f"<em>🔧 System: {msg['content']}</em>"])
            else:
                history.append([None, msg["content"]])
        return history

    def get_conversation_list(self):
        conversations = []
        for chat_id, data in self.conversations.items():
            conversations.append({
                "id": chat_id,
                "title": data["title"],
                "created_at": data["created_at"],
                "message_count": len(data["messages"])
            })
        conversations.sort(key=lambda x: x["created_at"], reverse=True)
        return conversations

    def delete_chat(self, chat_id):
        if chat_id in self.conversations:
            del self.conversations[chat_id]
            if self.current_chat_id == chat_id:
                if self.conversations:
                    # Switch to the most recent chat
                    self.current_chat_id = self.get_conversation_list()[0]['id']
                else:
                    self.create_new_chat()

# Initialize chat manager and global state
chat_manager = ChatManager()
app_state = { "uploaded_files": {}, "processing": False }

# --- Backend Interaction Functions ---

def get_document_list():
    doc_list = backend.get_display_list(backend.master_list)
    return list(doc_list.values())

def upload_file(files, progress=gr.Progress()):
    if not files:
        return gr.update(), "No files selected", gr.update()

    progress(0, desc="Starting upload...")
    new_files = []
    for i, file_obj in enumerate(files):
        filename = os.path.basename(file_obj.name)
        progress((i + 1) / len(files), desc=f"Processing {filename}...")
        if filename not in app_state["uploaded_files"]:
            if backend.add_pdf_file(file_obj.name):
                app_state["uploaded_files"][filename] = file_obj.name
                new_files.append(filename)

    status_msg = f"✅ Uploaded: {', '.join(new_files)}" if new_files else "ℹ️ Files already processed or failed."
    chat_manager.add_message("system", status_msg)
    return gr.update(choices=get_document_list()), status_msg, str(len(app_state["uploaded_files"]))

def update_document_selection(choices):
    doc_map = backend.get_display_list(backend.master_list)
    reverse_mapping = {v: k for k, v in doc_map.items()}
    chat = chat_manager.get_current_chat()
    chat["selected_docs"] = [reverse_mapping[choice] for choice in choices if choice in reverse_mapping]
    return f"📄 Selected: {len(choices)} documents"

def start_session():
    chat = chat_manager.get_current_chat()
    if not chat["selected_docs"]:
        return "❌ Please select at least one document.", chat_manager.get_chat_history_for_display(), get_session_status()

    try:
        if backend.load_docling_documents(chat["selected_docs"]):
            chat["rag_active"] = True
            msg = f"🚀 RAG session started with {len(chat['selected_docs'])} documents."
            chat_manager.add_message("system", msg)

            if chat["title"].startswith("New Chat"):
                doc_names = [backend.get_display_list(backend.master_list)[idx] for idx in chat["selected_docs"]]
                title = f"Chat: {doc_names[0]}"
                if len(doc_names) > 1:
                    title += f" & {len(doc_names)-1} more"
                chat_manager.update_chat_title(title[:50])

            return "🟢 RAG session started successfully", chat_manager.get_chat_history_for_display(), get_session_status()
    except Exception as e:
        error_msg = f"❌ Failed to load documents: {e}"
        chat_manager.add_message("system", error_msg)
        return error_msg, chat_manager.get_chat_history_for_display(), get_session_status()

def stop_session():
    chat = chat_manager.get_current_chat()
    chat["rag_active"] = False
    chat_manager.add_message("system", "⏹️ RAG session stopped")
    return "🔴 RAG session stopped", chat_manager.get_chat_history_for_display(), get_session_status()

def get_session_status():
    chat = chat_manager.get_current_chat()
    return "🟢 RAG Active" if chat["rag_active"] else "🔴 RAG Inactive"

def process_question(question):
    if not question.strip():
        return chat_manager.get_chat_history_for_display(), ""

    chat_manager.add_message("user", question)
    chat = chat_manager.get_current_chat()

    if not chat["rag_active"]:
        chat_manager.add_message("assistant", "⚠️ Please start the RAG session first.")
    else:
        try:
            response = backend.invoke_rag(question)
            formatted_response = format_assistant_response(response)
            chat_manager.add_message("assistant", formatted_response)
        except Exception as e:
            chat_manager.add_message("assistant", f"❌ Error: {e}")

    return chat_manager.get_chat_history_for_display(), ""

def format_assistant_response(response):
    if not response:
        return "❌ No response generated"

    response_text = response.strip()
    separator = "\n\n---------------------------\n\n"
    parts = response_text.split(separator)
    main_answer = parts[0].strip()
    details_html = ""

    if len(parts) > 1:
        details_content = parts[1].strip()
        details_html = f"""
<details>
<summary><strong>📚 View Sources & Evaluation</strong></summary>
<pre><code>{details_content}</code></pre>
</details>
"""
    return f"{main_answer}{details_html}"

# --- Conversation Management Functions ---

def create_new_conversation():
    chat_manager.create_new_chat()
    return (
        [], get_session_status(),
        gr.update(choices=get_document_list(), value=[]),
        "📄 Selected: 0 documents",
        get_conversation_dropdown()
    )

def get_conversation_dropdown():
    convos = chat_manager.get_conversation_list()
    choices = [(f"{c['title']} ({c['message_count']} msgs)", c['id']) for c in convos]
    return gr.update(choices=choices, value=chat_manager.current_chat_id)

def switch_conversation(chat_id):
    if chat_id and chat_manager.switch_to_chat(chat_id):
        chat = chat_manager.get_current_chat()
        doc_map = backend.get_display_list(backend.master_list)
        selected_doc_names = [doc_map[idx] for idx in chat["selected_docs"] if idx in doc_map]
        return (
            chat_manager.get_chat_history_for_display(),
            get_session_status(),
            gr.update(value=selected_doc_names),
            f"📄 Selected: {len(selected_doc_names)} documents"
        )
    return gr.update(), gr.update(), gr.update(), gr.update()

def delete_current_conversation():
    current_id = chat_manager.current_chat_id
    if current_id and len(chat_manager.conversations) > 1:
        chat_manager.delete_chat(current_id)
        # After deletion, switch to the new current chat
        return switch_conversation(chat_manager.current_chat_id) + (get_conversation_dropdown(),)
    else:
        # Don't delete the last chat, just clear it or notify user
        gr.Info("Cannot delete the last conversation.")
        return gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

# --- Gradio UI Definition ---

custom_css = """
.gradio-container { font-family: 'Segoe UI', sans-serif; }
.start-session-btn { background: linear-gradient(45deg, #00b894, #00cec9) !important; color: white !important; }
.stop-session-btn { background: linear-gradient(45deg, #d63031, #e17055) !important; color: white !important; }
.primary-button { background: linear-gradient(45deg, #6c5ce7, #a29bfe) !important; color: white !important; }
details { margin: 10px 0; border: 1px solid #ddd; border-radius: 8px; padding: 10px; background-color: #f9f9f9; }
details summary { cursor: pointer; font-weight: bold; }
"""

def create_interface():
    with gr.Blocks(title="Enhanced RAG Chat", css=custom_css) as demo:
        gr.Markdown("# 🤖 Enhanced RAG Chat Interface\n### Ask intelligent questions about your documents.")

        with gr.Row():
            with gr.Column(scale=1, min_width=350):
                gr.Markdown("## 📂 Document Management")
                with gr.Group():
                    gr.Markdown("### 📤 Upload Documents")
                    file_upload = gr.Files(label="Upload PDFs", file_types=[".pdf"], file_count="multiple", height=100)
                    upload_status = gr.Markdown("ℹ️ Ready to upload")

                with gr.Group():
                    gr.Markdown("### 📋 Select Documents for Session")
                    doc_list = gr.CheckboxGroup(label="Available Documents", choices=get_document_list(), interactive=True)
                    selection_status = gr.Markdown("📄 Selected: 0 documents")

                with gr.Group():
                    gr.Markdown("### 🎮 Session Controls")
                    with gr.Row():
                        start_btn = gr.Button("▶️ Start Session", elem_classes="start-session-btn")
                        stop_btn = gr.Button("⏹️ Stop Session", elem_classes="stop-session-btn")
                    session_status = gr.Markdown(get_session_status())

                with gr.Group():
                    gr.Markdown("### 💬 Chat History")
                    conversation_dropdown = gr.Dropdown(label="Select Conversation", interactive=True)
                    with gr.Row():
                        new_chat_btn = gr.Button("➕ New Chat", elem_classes="primary-button", scale=2)
                        delete_chat_btn = gr.Button("🗑️ Delete", elem_classes="stop-session-btn", scale=1)

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=600, bubble_full_width=False)
                with gr.Row():
                    question_input = gr.Textbox(placeholder="Ask anything about your documents...", show_label=False, container=False, scale=7)
                    submit_btn = gr.Button("Send", elem_classes="primary-button", scale=1)

        # --- Event Handlers ---
        file_upload.upload(upload_file, inputs=[file_upload], outputs=[doc_list, upload_status], show_progress="full")
        doc_list.change(update_document_selection, inputs=[doc_list], outputs=[selection_status])

        start_btn.click(start_session, outputs=[session_status, chatbot, session_status], show_progress="full")
        stop_btn.click(stop_session, outputs=[session_status, chatbot, session_status])

        new_chat_btn.click(create_new_conversation, outputs=[chatbot, session_status, doc_list, selection_status, conversation_dropdown])
        delete_chat_btn.click(delete_current_conversation, outputs=[chatbot, session_status, doc_list, selection_status, conversation_dropdown])
        conversation_dropdown.change(switch_conversation, inputs=[conversation_dropdown], outputs=[chatbot, session_is_active, doc_list, selection_status])

        question_input.submit(process_question, inputs=[question_input], outputs=[chatbot, question_input])
        submit_btn.click(process_question, inputs=[question_input], outputs=[chatbot, question_input])

        demo.load(lambda: (chat_manager.get_chat_history_for_display(), get_conversation_dropdown()), outputs=[chatbot, conversation_dropdown])

    return demo

if __name__ == "__main__":
    app_interface = create_interface()
    app_interface.launch(server_name="127.0.0.1", server_port=7860, debug=True)