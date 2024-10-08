from flask import Flask, render_template_string, request, jsonify
import requests
import logging
import threading

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
TAGS_ENDPOINT = f"{OLLAMA_API_BASE}/tags"
PULL_ENDPOINT = f"{OLLAMA_API_BASE}/pull"

# Replace with your actual Hugging Face token
HUGGING_FACE_TOKEN = "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"

# HTML template for the /egg route
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Egg Page</title>
    <style>
        body {
            background-color: black;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            color: white;
            font-family: Arial, sans-serif;
            padding: 10px;
            box-sizing: border-box;
        }
        .emoji {
            font-size: 60px;
            margin-bottom: 10px;
        }
        .controls {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            width: 100%;
            max-width: 600px;
        }
        .dropdown {
            padding: 5px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            flex: 1;
            margin-right: 5px;
        }
        .add-button {
            padding: 5px 10px;
            font-size: 18px;
            border: none;
            background-color: #4CAF50;
            color: white;
            cursor: pointer;
            border-radius: 4px;
        }
        .add-button:hover {
            background-color: #45a049;
        }
        .add-model-container {
            display: none;
            margin-bottom: 10px;
            width: 100%;
            max-width: 600px;
            display: flex;
        }
        .add-model-input {
            padding: 5px;
            font-size: 14px;
            border: none;
            border-radius: 4px 0 0 4px;
            width: 80%;
        }
        .add-model-submit {
            padding: 5px;
            font-size: 14px;
            border: none;
            background-color: #008CBA;
            color: white;
            cursor: pointer;
            border-radius: 0 4px 4px 0;
            width: 20%;
        }
        .add-model-submit:hover {
            background-color: #007BB5;
        }
        .toggle-container {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            width: 100%;
            max-width: 600px;
        }
        .toggle-label {
            margin-right: 5px;
            font-size: 14px;
        }
        .toggle-switch {
            position: relative;
            width: 40px;
            height: 20px;
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
            border-radius: 20px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 14px;
            width: 14px;
            left: 3px;
            bottom: 3px;
            background-color: white;
            transition: .4s;
            border-radius: 50%;
        }
        .toggle-switch input:checked + .slider {
            background-color: #2196F3;
        }
        .toggle-switch input:checked + .slider:before {
            transform: translateX(20px);
        }
        .input-container {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
            width: 100%;
            max-width: 600px;
        }
        .system-prompt-box {
            padding: 5px;
            font-size: 14px;
            border: none;
            border-radius: 4px;
            width: 150px;
            margin-right: 5px;
        }
        .input-box {
            padding: 5px;
            font-size: 14px;
            border: none;
            border-radius: 4px 0 0 4px;
            flex: 1;
        }
        .submit-button {
            padding: 5px;
            font-size: 14px;
            border: none;
            background-color: white;
            color: black;
            cursor: pointer;
            border-radius: 0 4px 4px 0;
            width: 30px;
        }
        .submit-button:hover {
            background-color: #e0e0e0;
        }
        .response-container {
            max-width: 600px;
            width: 100%;
            text-align: left;
            white-space: pre-wrap;
            margin-top: 10px;
            padding: 10px;
            border: 2px solid white;
            border-radius: 8px;
            background-color: #1a1a1a;
            max-height: 80vh;
            overflow-y: auto;
        }
        .loading {
            color: #00ff00;
        }
        .error {
            color: #ff0000;
        }
    </style>
</head>
<body>
    <div class="emoji">🥚</div>
    
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
    
    <!-- Toggle Switch for JSON Responses -->
    <div class="toggle-container">
        <span class="toggle-label">JSON Response:</span>
        <label class="toggle-switch">
            <input type="checkbox" id="jsonToggle" checked>
            <span class="slider"></span>
        </label>
    </div>
    
    <!-- System Prompt and Send Button -->
    <div class="input-container">
        <input type="text" id="systemPrompt" class="system-prompt-box" placeholder="System Prompt...">
        <input type="text" id="userInput" class="input-box" placeholder="Type your message here...">
        <button id="sendButton" class="submit-button">➡️</button>
    </div>
    
    <!-- Response Display -->
    <div id="response" class="response-container"></div>

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
            const responseDiv = document.getElementById('response');

            if (newModelName === "") {
                alert("Please enter a model name.");
                return;
            }

            // Clear previous response and show loading message
            responseDiv.innerText = "Pulling model '" + newModelName + "'...";
            responseDiv.classList.add('loading');
            responseDiv.classList.remove('error');

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
                responseDiv.classList.remove('loading');
                if (data.success) {
                    responseDiv.innerText = "Model '" + newModelName + "' pulled successfully.";
                    // Refresh the dropdown to include the new model
                    loadModels();
                    // Hide the add model input
                    document.getElementById('addModelContainer').style.display = 'none';
                    newModelInput.value = '';
                } else {
                    responseDiv.innerText = "Error: " + data.error;
                    responseDiv.classList.add('error');
                }
                // Re-enable add model inputs
                document.getElementById('newModelInput').disabled = false;
                document.getElementById('addModelSubmit').disabled = false;
            })
            .catch((error) => {
                console.error('Error pulling model:', error);
                responseDiv.classList.remove('loading');
                responseDiv.innerText = "An unexpected error occurred while pulling the model.";
                responseDiv.classList.add('error');
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
            const responseDiv = document.getElementById('response');

            if (userInput === "") {
                alert("Please enter a message.");
                return;
            }

            if (!selectedModel) {
                alert("Please select a model.");
                return;
            }

            // Clear previous response and show loading message
            responseDiv.innerText = "Processing...";
            responseDiv.classList.add('loading');
            responseDiv.classList.remove('error');

            // Disable inputs to prevent multiple submissions
            document.getElementById('userInput').disabled = true;
            document.getElementById('sendButton').disabled = true;
            document.getElementById('modelSelect').disabled = true;
            document.getElementById('systemPrompt').disabled = true;
            document.getElementById('jsonToggle').disabled = true;

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
                    json_response: jsonToggle
                }),
            })
            .then(response => response.json())
            .then(data => {
                responseDiv.classList.remove('loading');
                if (data.success) {
                    responseDiv.innerText = data.reply;
                } else {
                    responseDiv.innerText = "Error: " + data.error;
                    responseDiv.classList.add('error');
                }
                // Re-enable inputs
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('modelSelect').disabled = false;
                document.getElementById('systemPrompt').disabled = false;
                document.getElementById('jsonToggle').disabled = false;
            })
            .catch((error) => {
                console.error('Error:', error);
                responseDiv.classList.remove('loading');
                responseDiv.innerText = "An unexpected error occurred.";
                responseDiv.classList.add('error');
                // Re-enable inputs
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                document.getElementById('modelSelect').disabled = false;
                document.getElementById('systemPrompt').disabled = false;
                document.getElementById('jsonToggle').disabled = false;
            });
        });

        // Allow pressing Enter to send the message or add a model
        document.getElementById('userInput').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                document.getElementById('sendButton').click();
            }
        });
        document.getElementById('newModelInput').addEventListener('keyup', function(event) {
            if (event.key === 'Enter') {
                document.getElementById('addModelSubmit').click();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/egg')
def egg():
    """Render the Egg web interface."""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/models', methods=['GET'])
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
def ollama_api():
    """
    Handle incoming messages and interact with the Ollama API.
    Steps:
    1. Validate incoming data.
    2. Check if the selected model is available locally.
    3. If not available, pull the model using the provided token.
    4. Send the user message to the Ollama API's generate endpoint.
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
    logger.info(f"Received message from user: '{user_message}' with model: '{selected_model}'")
    logger.info(f"System prompt: '{system_prompt}'")
    logger.info(f"JSON response requested: {json_response}")

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

        # Step 4: Send the user message to the generate endpoint
        generate_payload = {
            'model': selected_model,
            'prompt': full_prompt,
            'stream': False,
            'format': "json" if json_response else ""
        }
        logger.info(f"Sending prompt to model '{selected_model}': '{full_prompt}' with JSON response={json_response}")
        generate_response = requests.post(GENERATE_ENDPOINT, json=generate_payload, headers={'Content-Type': 'application/json'}, timeout=30)

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
    # Run the Flask app on port 80 to make it accessible via http://localhost/egg
    # Note: Running on port 80 requires administrative privileges.
    # If you encounter a PermissionError, consider the following options:
    # 1. Run the script with sudo:
    #    sudo python3 egg_route.py
    # 2. Use a higher, non-privileged port (e.g., 5000) and configure a reverse proxy (e.g., Nginx).
    try:
        app.run(host='0.0.0.0', port=80)
    except PermissionError:
        logger.error("Permission denied: Unable to bind to port 80. Try running the script with sudo or use a higher port.")
