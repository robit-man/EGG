from flask import Flask, render_template_string, request, jsonify, redirect, url_for, session, flash
import requests
import logging
import threading
import os
from functools import wraps
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)

# Configure logging to output to console with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration Variables
OLLAMA_API_BASE = "http://localhost:11434/api"
GENERATE_ENDPOINT = f"{OLLAMA_API_BASE}/generate"
CHAT_ENDPOINT = f"{OLLAMA_API_BASE}/chat"
TAGS_ENDPOINT = f"{OLLAMA_API_BASE}/tags"
PULL_ENDPOINT = f"{OLLAMA_API_BASE}/pull"

# Load secrets from environment variables
app.secret_key = os.getenv('SECRET_KEY', 'default_secret_key')  # Replace with a strong secret key
ACCESS_PASSWORD = os.getenv('ACCESS_PASSWORD', 'password')  # Replace with a secure password

# HTML templates
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Login</title>
    <style>
        body {
            background-color: #121212;
            color: white;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            height: 100vh;
            align-items: center;
            justify-content: center;
            margin: 0;
        }
        .login-container {
            background-color: #1e1e1e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
            width: 90%;
            max-width: 400px;
        }
        .login-container h2 {
            text-align: center;
            margin-bottom: 20px;
            font-size: 24px;
        }
        .login-container input[type="password"] {
            width: 100%;
            padding: 12px;
            margin-bottom: 15px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            background-color: #2c2c2c;
            color: white;
        }
        .login-container button {
            width: 100%;
            padding: 12px;
            background-color: #4CAF50;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            color: white;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        .login-container button:hover {
            background-color: #45a049;
        }
        .error-message {
            color: #ff4d4d;
            text-align: center;
            margin-bottom: 10px;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="login-container">
        <h2>Secure Login</h2>
        {% with messages = get_flashed_messages() %}
          {% if messages %}
            <div class="error-message">{{ messages[0] }}</div>
          {% endif %}
        {% endwith %}
        <form method="POST" action="{{ url_for('login') }}">
            <input type="password" name="password" placeholder="Enter Password" required>
            <button type="submit">Login</button>
        </form>
    </div>
</body>
</html>
"""

EGG_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Egg Chat Interface</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            background-color: #121212;
            color: #e0e0e0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0;
            padding: 10px;
            box-sizing: border-box;
            height: 100vh;
        }
        .header {
            width: 100%;
            max-width: 600px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .emoji {
            font-size: 40px;
        }
        .logout-button {
            padding: 8px 16px;
            font-size: 14px;
            border: none;
            background-color: #f44336;
            color: white;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .logout-button:hover {
            background-color: #da190b;
        }
        .controls, .toggle-container {
            width: 100%;
            max-width: 600px;
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .controls {
            justify-content: space-between;
        }
        .dropdown {
            flex: 1;
            padding: 10px;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            background-color: #2c2c2c;
            color: white;
            margin-right: 10px;
        }
        .add-button {
            padding: 10px 16px;
            font-size: 18px;
            border: none;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            border-radius: 8px;
            transition: background-color 0.3s ease;
        }
        .add-button:hover {
            background-color: #45a049;
        }
        .add-model-container {
            display: none;
            width: 100%;
            max-width: 600px;
            margin-bottom: 10px;
        }
        .add-model-container input {
            width: 80%;
            padding: 10px;
            font-size: 16px;
            border: none;
            border-radius: 8px 0 0 8px;
            background-color: #2c2c2c;
            color: white;
        }
        .add-model-container button {
            width: 20%;
            padding: 10px;
            font-size: 16px;
            border: none;
            background-color: #008CBA;
            color: white;
            cursor: pointer;
            border-radius: 0 8px 8px 0;
            transition: background-color 0.3s ease;
        }
        .add-model-container button:hover {
            background-color: #007BB5;
        }
        .toggle-container {
            justify-content: flex-start;
        }
        .toggle-label {
            margin-right: 10px;
            font-size: 16px;
        }
        .toggle-switch {
            position: relative;
            width: 50px;
            height: 24px;
        }
        .toggle-switch input {
            opacity: 0;
            width: 0;
            height: 0;
        }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background-color: #ccc;
            transition: .4s;
            border-radius: 24px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 16px;
            width: 16px;
            left: 4px;
            bottom: 4px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        .toggle-switch input:checked + .slider {
            background-color: #2196F3;
        }
        .toggle-switch input:checked + .slider:before {
            transform: translateX(26px);
        }
        .input-container {
            width: 100%;
            max-width: 600px;
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .system-prompt-box {
            width: 150px;
            padding: 10px;
            font-size: 16px;
            border: none;
            border-radius: 8px 0 0 8px;
            background-color: #2c2c2c;
            color: white;
            margin-right: 10px;
        }
        .input-box {
            flex: 1;
            padding: 10px;
            font-size: 16px;
            border: none;
            border-radius: 0 8px 8px 0;
            background-color: #2c2c2c;
            color: white;
        }
        .submit-button {
            display: none; /* Hidden as we will handle sending via Enter key */
        }
        .chat-container {
            width: 100%;
            max-width: 600px;
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow-y: auto;
            padding-bottom: 10px;
        }
        .message {
            margin-bottom: 10px;
            display: flex;
            flex-direction: column;
        }
        .user-message {
            align-self: flex-end;
            background-color: #008CBA;
            color: white;
            padding: 10px;
            border-radius: 12px 12px 0 12px;
            max-width: 80%;
            word-wrap: break-word;
        }
        .bot-message {
            align-self: flex-start;
            background-color: #4CAF50;
            color: white;
            padding: 10px;
            border-radius: 12px 12px 12px 0;
            max-width: 80%;
            word-wrap: break-word;
        }
        .loading {
            color: #00ff00;
            font-style: italic;
        }
        .error {
            color: #ff4d4d;
            font-style: italic;
        }
        @media (max-width: 600px) {
            .system-prompt-box {
                width: 100px;
            }
            .toggle-label {
                font-size: 14px;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="emoji">🥚</div>
        <form method="POST" action="{{ url_for('logout') }}">
            <button type="submit" class="logout-button">Logout</button>
        </form>
    </div>
    
    <!-- Controls: Dropdown and Add Button -->
    <div class="controls">
        <select id="modelSelect" class="dropdown">
            <option value="" disabled selected>Loading models...</option>
        </select>
        <button id="addModelButton" class="add-button">+</button>
    </div>
    
    <!-- Add Model Input -->
    <div id="addModelContainer" class="add-model-container">
        <input type="text" id="newModelInput" class="add-model-input" placeholder="Enter model name...">
        <button id="addModelSubmit" class="add-model-submit">➡️</button>
    </div>
    
    <!-- Toggle Switches: JSON Response and API Mode -->
    <div class="toggle-container">
        <span class="toggle-label">JSON Response:</span>
        <label class="toggle-switch">
            <input type="checkbox" id="jsonToggle" checked>
            <span class="slider"></span>
        </label>
    </div>
    
    <div class="toggle-container">
        <span class="toggle-label">API Mode:</span>
        <label class="toggle-switch">
            <input type="checkbox" id="apiModeToggle" checked>
            <span class="slider"></span>
        </label>
    </div>
    
    <!-- System Prompt and Send Button -->
    <div class="input-container">
        <input type="text" id="systemPrompt" class="system-prompt-box" placeholder="System Prompt...">
        <input type="text" id="userInput" class="input-box" placeholder="Type your message here..." autocomplete="off">
        <button id="sendButton" class="submit-button">Send</button>
    </div>
    
    <!-- Chat Container -->
    <div id="chatContainer" class="chat-container"></div>

    <script>
        // Function to fetch and populate the dropdown with available models
        function loadModels() {
            fetch('/api/models')
                .then(response => response.json())
                .then(data => {
                    const modelSelect = document.getElementById('modelSelect');
                    modelSelect.innerHTML = ''; // Clear existing options
                    data.models.forEach(model => {
                        const option = document.createElement('option');
                        option.value = model;
                        option.text = model;
                        modelSelect.appendChild(option);
                    });
                    // Set default selections if available
                    const defaultModels = ['llama3.2:3b', 'llama3.2:3b-instruct-fp16'];
                    defaultModels.forEach(defaultModel => {
                        for (let i = 0; i < modelSelect.options.length; i++) {
                            if (modelSelect.options[i].value === defaultModel) {
                                modelSelect.options[i].selected = true;
                                break;
                            }
                        }
                    });
                })
                .catch(error => {
                    console.error('Error fetching models:', error);
                    const modelSelect = document.getElementById('modelSelect');
                    modelSelect.innerHTML = '<option value="" disabled selected>Failed to load models</option>';
                });
        }

        // Load models on page load
        window.onload = loadModels;

        // Handle Add Model Button Click
        document.getElementById('addModelButton').addEventListener('click', function() {
            const addModelContainer = document.getElementById('addModelContainer');
            if (addModelContainer.style.display === 'none' || addModelContainer.style.display === '') {
                addModelContainer.style.display = 'flex';
            } else {
                addModelContainer.style.display = 'none';
            }
        });

        // Handle Add Model Submit
        document.getElementById('addModelSubmit').addEventListener('click', function() {
            const newModelInput = document.getElementById('newModelInput');
            const newModelName = newModelInput.value.trim();
            const chatContainer = document.getElementById('chatContainer');

            if (newModelName === "") {
                alert("Please enter a model name.");
                return;
            }

            // Append system message
            appendMessage('system', "Pulling model '" + newModelName + "'...", true);

            // Disable add model inputs to prevent multiple submissions
            document.getElementById('newModelInput').disabled = true;
            document.getElementById('addModelSubmit').disabled = true;

            // Send the new model name to the Flask backend to pull the model
            fetch('/api/pull_model', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ model: newModelName }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    appendMessage('system', "Model '" + newModelName + "' pulled successfully.", false);
                    // Refresh the dropdown to include the new model
                    loadModels();
                    // Hide the add model input
                    document.getElementById('addModelContainer').style.display = 'none';
                    newModelInput.value = '';
                } else {
                    appendMessage('error', "Error: " + data.error, false);
                }
                // Re-enable add model inputs
                document.getElementById('newModelInput').disabled = false;
                document.getElementById('addModelSubmit').disabled = false;
            })
            .catch((error) => {
                console.error('Error pulling model:', error);
                appendMessage('error', "An unexpected error occurred while pulling the model.", false);
                // Re-enable add model inputs
                document.getElementById('newModelInput').disabled = false;
                document.getElementById('addModelSubmit').disabled = false;
            });
        });

        // Handle Send Button Click
        document.getElementById('sendButton').addEventListener('click', function() {
            const userInput = document.getElementById('userInput').value.trim();
            const selectedModel = document.getElementById('modelSelect').value;
            const systemPrompt = document.getElementById('systemPrompt').value.trim();
            const jsonToggle = document.getElementById('jsonToggle').checked;
            const apiModeToggle = document.getElementById('apiModeToggle').checked;
            const chatContainer = document.getElementById('chatContainer');

            if (userInput === "") {
                alert("Please enter a message.");
                return;
            }

            if (!selectedModel) {
                alert("Please select a model.");
                return;
            }

            // Append user message
            appendMessage('user', userInput, false);

            // Clear input box
            document.getElementById('userInput').value = '';

            // Append system message
            appendMessage('system', "Processing...", true);

            // Disable inputs to prevent multiple submissions
            document.getElementById('userInput').disabled = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('modelSelect').disabled = true;
            document.getElementById('systemPrompt').disabled = true;
            document.getElementById('jsonToggle').disabled = true;
            document.getElementById('apiModeToggle').disabled = true;

            // Send the input to the Flask backend
            fetch('/api/ollama', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ 
                    model: selectedModel, 
                    message: userInput,
                    system_prompt: systemPrompt,
                    json_response: jsonToggle,
                    api_mode: apiModeToggle
                }),
            })
            .then(response => response.json())
            .then(data => {
                // Remove "Processing..." system message
                removeLastMessage();

                if (data.success) {
                    appendMessage('bot', data.reply, false);
                } else {
                    appendMessage('error', "Error: " + data.error, false);
                }
                // Re-enable inputs
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('modelSelect').disabled = false;
                document.getElementById('systemPrompt').disabled = false;
                document.getElementById('jsonToggle').disabled = false;
                document.getElementById('apiModeToggle').disabled = false;
            })
            .catch((error) => {
                console.error('Error:', error);
                // Remove "Processing..." system message
                removeLastMessage();
                appendMessage('error', "An unexpected error occurred.", false);
                // Re-enable inputs
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('modelSelect').disabled = false;
                document.getElementById('systemPrompt').disabled = false;
                document.getElementById('jsonToggle').disabled = false;
                document.getElementById('apiModeToggle').disabled = false;
            });
        });

        // Allow pressing Enter to send the message or add a model
        document.getElementById('userInput').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                document.getElementById('sendButton').click();
            }
        });
        document.getElementById('newModelInput').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                event.preventDefault();
                document.getElementById('addModelSubmit').click();
            }
        });

        // Function to append messages to the chat container
        function appendMessage(sender, message, isLoading) {
            const chatContainer = document.getElementById('chatContainer');
            const messageDiv = document.createElement('div');
            messageDiv.classList.add('message');

            if (sender === 'user') {
                messageDiv.classList.add('user-message');
            } else if (sender === 'bot') {
                messageDiv.classList.add('bot-message');
            } else if (sender === 'system') {
                messageDiv.style.alignSelf = 'center';
                messageDiv.style.backgroundColor = '#444444';
                messageDiv.style.color = '#ffffff';
                messageDiv.style.borderRadius = '12px';
            } else if (sender === 'error') {
                messageDiv.style.alignSelf = 'center';
                messageDiv.style.color = '#ff4d4d';
                messageDiv.style.fontStyle = 'italic';
            }

            messageDiv.innerText = message;
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        // Function to remove the last message (used to remove "Processing...")
        function removeLastMessage() {
            const chatContainer = document.getElementById('chatContainer');
            const lastMessage = chatContainer.lastElementChild;
            if (lastMessage && lastMessage.classList.contains('message') && lastMessage.innerText === "Processing...") {
                chatContainer.removeChild(lastMessage);
            }
        }
    </script>
</body>
</html>
"""

# Authentication Decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Routes

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Render the login page and handle login logic."""
    if request.method == 'POST':
        entered_password = request.form.get('password')
        if entered_password == ACCESS_PASSWORD:
            session['logged_in'] = True
            logger.info("User logged in successfully.")
            return redirect(url_for('egg'))
        else:
            flash("Invalid password. Please try again.")
            logger.warning("Failed login attempt.")
            return render_template_string(LOGIN_TEMPLATE)
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/logout', methods=['POST'])
@login_required
def logout():
    """Handle user logout."""
    session.pop('logged_in', None)
    logger.info("User logged out.")
    return redirect(url_for('login'))

@app.route('/egg')
@login_required
def egg():
    """Render the Egg web interface."""
    return render_template_string(EGG_TEMPLATE)

@app.route('/api/models', methods=['GET'])
@login_required
def get_models():
    """Fetch and return the list of available models from Ollama API."""
    try:
        response = requests.get(TAGS_ENDPOINT, timeout=30)
        if response.status_code == 200:
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            logger.info(f"Available models: {models}")
            return jsonify({'models': models}), 200
        else:
            logger.error(f"Failed to retrieve models from Ollama API: {response.text}")
            return jsonify({'success': False, 'error': 'Failed to retrieve models from Ollama API.'}), 500
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from Ollama API: {e}")
        return jsonify({'success': False, 'error': 'Error fetching models from Ollama API.'}), 500

def pull_model_thread(model_name, response_queue):
    """
    Thread function to pull a model without blocking the main thread.
    """
    try:
        pull_payload = {'name': model_name}
        pull_headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {HUGGING_FACE_TOKEN}'
        }
        logger.info(f"Sending pull request for model '{model_name}'")
        pull_response = requests.post(PULL_ENDPOINT, json=pull_payload, headers=pull_headers, timeout=600)  # Increased timeout for model pulling

        if pull_response.status_code == 200:
            logger.info(f"Model '{model_name}' pulled successfully.")
            response_queue['success'] = True
            response_queue['message'] = f"Model '{model_name}' pulled successfully."
        else:
            try:
                error_message = pull_response.json().get('error', 'Unknown error during model pull.')
            except ValueError:
                error_message = 'Unknown error during model pull.'
            logger.error(f"Failed to pull model '{model_name}': {error_message}")
            response_queue['success'] = False
            response_queue['message'] = f"Failed to pull model '{model_name}': {error_message}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error pulling model '{model_name}': {e}")
        response_queue['success'] = False
        response_queue['message'] = f"Error pulling model '{model_name}': {e}"

@app.route('/api/pull_model', methods=['POST'])
@login_required
def pull_model():
    """
    Handle pulling a new model. This endpoint initiates the model pulling process.
    """
    data = request.get_json()
    if not data or 'model' not in data:
        logger.error("Invalid request data. 'model' is required.")
        return jsonify({'success': False, 'error': 'Invalid request data. "model" is required.'}), 400

    model_name = data['model'].strip()
    if model_name == "":
        logger.error("Model name cannot be empty.")
        return jsonify({'success': False, 'error': 'Model name cannot be empty.'}), 400

    # Check if the model is already available
    try:
        tags_response = requests.get(TAGS_ENDPOINT, timeout=30)
        if tags_response.status_code != 200:
            logger.error(f"Failed to retrieve models from Ollama API: {tags_response.text}")
            return jsonify({'success': False, 'error': 'Failed to retrieve models from Ollama API.'}), 500

        available_models = [model['name'] for model in tags_response.json().get('models', [])]
        logger.info(f"Available models before pull: {available_models}")

        if model_name in available_models:
            logger.info(f"Model '{model_name}' is already available.")
            return jsonify({'success': True, 'message': f"Model '{model_name}' is already available."}), 200

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching models from Ollama API: {e}")
        return jsonify({'success': False, 'error': 'Error fetching models from Ollama API.'}), 500

    # Initiate model pulling in a separate thread to avoid blocking
    response_queue = {}
    thread = threading.Thread(target=pull_model_thread, args=(model_name, response_queue))
    thread.start()

    # Wait for the thread to finish (could be replaced with asynchronous handling)
    thread.join()

    if response_queue.get('success'):
        return jsonify({'success': True, 'message': response_queue.get('message')}), 200
    else:
        return jsonify({'success': False, 'error': response_queue.get('message')}), 500

@app.route('/api/ollama', methods=['POST'])
@login_required
def ollama_api():
    """
    Handle incoming messages and interact with the Ollama API.
    Steps:
    1. Validate incoming data.
    2. Check if the selected model is available locally.
    3. If not available, pull the model using the provided token.
    4. Send the user message to the Ollama API's appropriate endpoint.
    5. Return the AI-generated response.
    """
    data = request.get_json()
    if not data or 'message' not in data or 'model' not in data:
        logger.error("Invalid request data. 'model' and 'message' are required.")
        return jsonify({'success': False, 'error': 'Invalid request data. "model" and "message" are required.'}), 400

    selected_model = data['model']
    user_message = data['message']
    system_prompt = data.get('system_prompt', '')
    json_response = data.get('json_response', True)
    api_mode = data.get('api_mode', True)  # True for Chat API, False for Generation API
    logger.info(f"Received message from user: '{user_message}' with model: '{selected_model}'")
    logger.info(f"System prompt: '{system_prompt}'")
    logger.info(f"JSON response requested: {json_response}")
    logger.info(f"API mode (Chat=True, Generation=False): {api_mode}")

    try:
        # Step 1: Check if the selected model is available
        logger.info(f"Checking availability of model '{selected_model}'")
        tags_response = requests.get(TAGS_ENDPOINT, timeout=30)

        if tags_response.status_code != 200:
            logger.error(f"Failed to retrieve models from Ollama API: {tags_response.text}")
            return jsonify({'success': False, 'error': 'Failed to retrieve models from Ollama API.'}), 500

        available_models = [model['name'] for model in tags_response.json().get('models', [])]
        logger.info(f"Available models: {available_models}")

        if selected_model not in available_models:
            logger.info(f"Model '{selected_model}' not found locally. Initiating pull.")
            # Step 2: Pull the model
            pull_payload = {'name': selected_model}
            pull_headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {HUGGING_FACE_TOKEN}'
            }
            logger.info(f"Sending pull request for model '{selected_model}'")
            pull_response = requests.post(PULL_ENDPOINT, json=pull_payload, headers=pull_headers, timeout=600)  # Increased timeout for model pulling

            if pull_response.status_code != 200:
                # Attempt to parse error message
                try:
                    error_message = pull_response.json().get('error', 'Unknown error during model pull.')
                except ValueError:
                    error_message = 'Unknown error during model pull.'
                logger.error(f"Failed to pull model '{selected_model}': {error_message}")
                return jsonify({'success': False, 'error': f"Failed to pull model '{selected_model}': {error_message}"}), 500

            logger.info(f"Model '{selected_model}' pulled successfully.")
            # Optionally, refresh the available models
            available_models.append(selected_model)

        # Step 3: Prepare the prompt
        if system_prompt:
            full_prompt = system_prompt + "\n" + user_message
        else:
            full_prompt = user_message

        # Step 4: Send the user message to the appropriate endpoint
        if api_mode:
            # Chat API
            endpoint = CHAT_ENDPOINT
            generate_payload = {
                'model': selected_model,
                'prompt': full_prompt,
                'stream': False,
                'format': "json" if json_response else ""
            }
        else:
            # Generation API
            endpoint = GENERATE_ENDPOINT
            generate_payload = {
                'model': selected_model,
                'prompt': full_prompt,
                'stream': False,
                'format': "json" if json_response else ""
            }

        logger.info(f"Sending prompt to model '{selected_model}' via {'Chat' if api_mode else 'Generation'} API: '{full_prompt}' with JSON response={json_response}")
        generate_response = requests.post(endpoint, json=generate_payload, headers={'Content-Type': 'application/json'}, timeout=30)

        logger.info(f"Ollama API responded with status code {generate_response.status_code}")
        logger.info(f"Ollama API response text: {generate_response.text}")

        if generate_response.status_code == 200:
            try:
                response_json = generate_response.json()
                reply = response_json.get('response', 'No reply from Ollama.')
                if not json_response:
                    # If JSON response is not requested, ensure reply is plain text
                    if isinstance(reply, str):
                        pass  # Already plain text
                    else:
                        reply = str(reply)
                logger.info(f"Received reply from Ollama: '{reply}'")
                return jsonify({'success': True, 'reply': reply}), 200
            except ValueError as ve:
                logger.error(f"JSON decoding failed: {ve}")
                logger.error(f"Raw response was: {generate_response.text}")
                return jsonify({'success': False, 'error': 'Invalid JSON response from Ollama.'}), 500
        else:
            try:
                error_message = generate_response.json().get('error', 'Unknown error from Ollama.')
                logger.error(f"Ollama API returned error: {error_message}")
                return jsonify({'success': False, 'error': error_message}), generate_response.status_code
            except ValueError:
                logger.error("JSON decoding failed for error response.")
                return jsonify({'success': False, 'error': 'Unknown error from Ollama.'}), generate_response.status_code

    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with Ollama API: {e}")
        return jsonify({'success': False, 'error': 'Error communicating with Ollama API.'}), 500

if __name__ == '__main__':
    """
    Run the Flask app on port 80 to make it accessible via http://localhost/egg
    Note: Running on port 80 requires administrative privileges.
    If you encounter a PermissionError, consider the following options:
    1. Run the script with sudo:
       sudo python3 egg_route.py
    2. Use a higher, non-privileged port (e.g., 5000) and configure a reverse proxy (e.g., Nginx).
    """
    try:
        app.run(host='0.0.0.0', port=80)
    except PermissionError:
        logger.error("Permission denied: Unable to bind to port 80. Try running the script with sudo or use a higher port.")
