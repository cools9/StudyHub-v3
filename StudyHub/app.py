from flask import Flask, render_template, request, redirect, url_for, abort, session, flash
from supabase import create_client, Client
import os
from functools import wraps
import random
import os

import api_keys
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

app = Flask(__name__, template_folder='templates')
app.secret_key = os.urandom(24)  # For session management

# Supabase URL and API Key
url = "https://xbsywxlzzvjkfnurmoaj.supabase.co"
key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inhic3l3eGx6enZqa2ZudXJtb2FqIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjMzNjExODgsImV4cCI6MjAzODkzNzE4OH0.fskAVBTCjzBJl5gpr08dhmgAn5y-oRJhPjAWkFL5iQ4"  # Replace with your actual API key

# Initialize Supabase client
supabase: Client = create_client(url, key)

# Custom login_required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash("Please log in to access this page.")
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# This route shows the newbie.html page
@app.route('/')
def newbie():
    return render_template('newbie.html')  # Initial page

# The main route (index.html) after login
@app.route('/index')
@login_required
def main():
    # Query the 'uploads' table
    response = supabase.table("uploads").select("*").execute()
    if response:
        data = response.data
    else:
        data = []

    return render_template('index.html', data=data)

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the user already exists
        response = supabase.table("users").select("*").eq('username', username).execute()
        if response.data:
            flash("Username already exists, please log in.")
            return redirect(url_for('login'))

        # Insert new user into Supabase
        response = supabase.table("users").insert({
            'username': username,
            'password': password
        }).execute()

        if response.data:
            session['username'] = username
            return redirect(url_for('main'))  # Redirect to index.html after signup
        else:
            error_message = response.error or "Error signing up"
            return f"{error_message}", 500

    return render_template('signup.html')

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Check if the username and password match
        response = supabase.table("users").select("*").eq('username', username).eq('password', password).execute()

        if response.data:
            session['username'] = username
            return redirect(url_for('main'))  # Redirect to index.html after login
        else:
            flash("Invalid username or password")
            return redirect(url_for('login'))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

# Add Notes route
@app.route('/add-notes', methods=['GET', 'POST'])
@login_required
def add_notes():
    if request.method == 'POST':
        title = request.form.get('title')
        description = request.form.get('description')
        content = request.form.get('content')
        uploaded_by = session.get('username')  # Get the currently logged-in user's username

        # Insert new entry into Supabase
        response = supabase.table("uploads").insert({
            'title': title,
            'description': description,
            'content': content,
            'uploaded_by': uploaded_by  # Add the username to the uploaded_by column
        }).execute()

        if response.data:
            return redirect(url_for('main'))
        else:
            error_message = response.error or "Error adding entry"
            return f"{error_message}", 500

    return render_template('add-notes.html')

# View specific content route
@app.route('/content/<int:content_id>')
@login_required
def content(content_id):
    # Query the 'uploads' table for a specific content ID
    response = supabase.table("uploads").select("*").eq('id', content_id).execute()

    if response and response.data:
        content = response.data[0]  # Assuming there's only one result
    else:
        abort(404)  # If not found, return a 404 error

    return render_template('content.html', content=content)

# My Posts route
@app.route('/my-posts')
@login_required
def my_posts():
    username = session.get('username')  # Get the currently logged-in user's username

    # Query the 'uploads' table for posts uploaded by the logged-in user
    response = supabase.table("uploads").select("*").eq('uploaded_by', username).execute()

    if response:
        data = response.data
    else:
        data = []

    return render_template('my_posts.html', data=data)

# Session info route
@app.route('/session')
@login_required
def session_info():
    return render_template('session.html')

# Create session route
def generate_session_code():
    return random.randint(100000, 999999)

@app.route('/create-session', methods=['GET', 'POST'])
@login_required
def create_session():
    if request.method == 'POST':
        session_name = request.form.get('session_name')
        session_code = generate_session_code()  # Generate a 6-digit random code
        created_by = session.get('username')  # Get the currently logged-in user's username

        # Insert the new session into the 'sessions' table in Supabase
        response = supabase.table("sessions").insert({
            'running_session': session_name,
            'session_code': session_code,
        }).execute()

        if response.data:
            flash_message = f"Session '{session_name}' created successfully with code: {session_code}"
            return render_template('create_session.html', flash_message=flash_message, session_code=session_code)
        else:
            error_message = response.error or "Error creating session"
            return f"{error_message}", 500

    return render_template('create_session.html')

@app.route('/send-message', methods=['POST'])
@login_required
def send_message():
    try:
        message = request.form.get('message')
        session_name = request.form.get('session_name')
        sent_by = session.get('username')  # Get the currently logged-in user's username

        if not message or not session_name:
            app.logger.error("Message or session name is missing.")
            return {"status": "error", "message": "Message and session name are required"}, 400

        # Check if the message starts with a comma to trigger AI response


        # Insert the user's message into the session_messages table
        response = supabase.table("session_messages").insert({
            'message': message,
            'session_name': session_name,
            'sent_by':sent_by
        }).execute()

        if message.startswith(","):
            user_question = message[1:].strip()  # Remove the comma and get the question
            ai_response = generate_ai_response(user_question)  # Function to get AI response

                    # Store the AI response in the session messages table
            response = supabase.table("session_messages").insert({
                'message': ai_response,
                'session_name': session_name,
                'sent_by':"ai"
            }).execute()

        if response.data:
            return {"status": "success"}, 200
        else:
            error_message = response.error or "Error sending message"
            app.logger.error(f"Error from Supabase: {error_message}")
            return {"status": "error", "message": error_message}, 500

    except Exception as e:
        app.logger.error(f"Unexpected error occurred while sending message: {e}")
        return {"status": "error", "message": "Unexpected error occurred"}, 500

# Store memory globally or in a higher scope so it persists across function calls
groq_api_key = api_keys.secret
model = 'llama3-8b-8192'
groq_chat = ChatGroq(
    groq_api_key=groq_api_key,
    model_name=model
)

# Global memory object to store conversation history
memory = ConversationBufferWindowMemory(k=32, memory_key="chat_history", return_messages=True)

def generate_ai_response(user_question):
    """
    Generates an AI response to the given user question using a persistent memory.
    """
    system_prompt = 'You are a friendly conversational chatbot'

    # Construct the chat prompt template using memory and current user input
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),  # Persistent system prompt
            MessagesPlaceholder(variable_name="chat_history"),  # Conversation history
            HumanMessagePromptTemplate.from_template("{human_input}"),  # User input
        ]
    )

    # Create the conversation chain using Groq LangChain and the persistent memory
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory  # Use the persistent memory
    )

    # Generate the AI response and return it
    response = conversation.predict(human_input=user_question)
    return response







# Join session route
@app.route('/join-session')
@login_required
def session_join():
    return render_template('join_session.html')

# Get messages for a session
@app.route('/get-messages', methods=['GET'])
@login_required
def get_messages():
    session_name = request.args.get('session_name')

    # Fetch messages for the specified session
    response = supabase.table("session_messages").select("*").eq('session_name', session_name).execute()

    if response.data:
        messages = response.data
        return {"messages": messages}, 200
    else:
        return {"messages": []}, 200

# Chat room route
@app.route('/chat-room')
@login_required
def chat_room():
    return render_template('chat_room.html')

# Join session by code route
@app.route('/join-session-code', methods=['POST'])
@login_required
def join_session_code():
    session_code = request.form.get('code')

    # Query the 'sessions' table to find a session with the provided code
    response = supabase.table("sessions").select("*").eq('session_code', session_code).execute()

    if response.data:
        session_name = response.data[0]['running_session']  # Get the session name
        return redirect(url_for('chat_room', session_name=session_name))
    else:
        # If no session found, return with an error message
        return render_template('join-session.html', error_message="Invalid session code. Please try again.")


"""
def run_flask():
    app.run(port=5000, debug=False, use_reloader=False)  # Run Flask app on port 5000

import webview.menu as wm

# Function for 'Undo' action
def undo_action():
    active_window = webview.active_window()
    if active_window:
        active_window.evaluate_js("window.history.back();")

# Function for 'Redo' action
def redo_action():
    active_window = webview.active_window()
    if active_window:
        active_window.evaluate_js("window.history.forward();")

if __name__ == '__main__':
    # Start Flask in a separate thread
    flask_thread = threading.Thread(target=run_flask)
    flask_thread.daemon = True
    flask_thread.start()

    # Open the Flask app in a pywebview window
    webview.create_window('StudyHub', 'http://127.0.0.1:5000')  # Load Flask URL
    menu_items = [
        wm.Menu(
            'Edit',
            [
                wm.MenuAction('Undo', undo_action),
                wm.MenuAction('Redo', redo_action),
            ],
        ),
    ]
    webview.start(menu=menu_items)

"""

if __name__ == '__main__':
    app.run(port=5000, debug=False, use_reloader=False)
