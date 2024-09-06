import argparse
import asyncio
import sys
import os
import termcolor
import importlib.util
from nano_llm import NanoLLM, ChatHistory, BotFunctions

SETTINGS_FILE = 'settings.txt'
LLM_FUNCTIONS_MODULE = 'llm_functions'  # Name of the module file (without .py)

class ChatManager:
    def __init__(self, model, system_prompt, max_input_tokens, max_new_tokens, temperature, repetition_penalty, top_p, input_port=None, output_port=None, output_ip="127.0.0.1"):
        self.chat_history = ChatHistory(model, system_prompt=system_prompt)
        self.model = model
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_p = top_p
        self.input_port = input_port
        self.output_port = output_port
        self.output_ip = output_ip

        self.load_settings()
        self.functions_module = self.load_functions_module()
        self.bot_functions = BotFunctions()

        # Set the initial system prompt
        self.modify_system_prompt(system_prompt)

    def load_settings(self):
        """Load settings from settings.txt if the file exists."""
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as file:
                lines = file.readlines()
                prompt_lines = []
                inside_prompt = False

                for line in lines:
                    line = line.strip()
                    if line == "BEGIN_PROMPT":
                        inside_prompt = True
                        continue
                    elif line == "END_PROMPT":
                        inside_prompt = False
                        self.system_prompt = "\n".join(prompt_lines)
                        prompt_lines = []
                    elif inside_prompt:
                        prompt_lines.append(line)
                    else:
                        key, value = line.split('=', 1)
                        if key == "max_input_tokens":
                            self.max_input_tokens = int(value)
                        elif key == "max_new_tokens":
                            self.max_new_tokens = int(value)
                        elif key == "temperature":
                            self.temperature = float(value)
                        elif key == "repetition_penalty":
                            self.repetition_penalty = float(value)
                        elif key == "top_p":
                            self.top_p = float(value)
                        elif key == "input_port":
                            self.input_port = int(value)
                        elif key == "output_port":
                            self.output_port = int(value)

    def save_settings(self):
        """Save the current settings to settings.txt."""
        with open(SETTINGS_FILE, 'w') as file:
            file.write(f"max_input_tokens={self.max_input_tokens}\n")
            file.write(f"max_new_tokens={self.max_new_tokens}\n")
            file.write(f"temperature={self.temperature}\n")
            file.write(f"repetition_penalty={self.repetition_penalty}\n")
            file.write(f"top_p={self.top_p}\n")
            if self.input_port:
                file.write(f"input_port={self.input_port}\n")
            if self.output_port:
                file.write(f"output_port={self.output_port}\n")
            if self.output_ip:
                file.write(f"output_ip={self.output_ip}\n")
            file.write("BEGIN_PROMPT\n")
            file.write(self.system_prompt + "\n")
            file.write("END_PROMPT\n")

    def load_functions_module(self):
        """Dynamically load the llm_functions module and register its functions."""
        spec = importlib.util.spec_from_file_location(LLM_FUNCTIONS_MODULE, f"{LLM_FUNCTIONS_MODULE}.py")
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        else:
            raise ImportError(f"Could not load the module {LLM_FUNCTIONS_MODULE}.py")

    def append_message(self, role, message):
        """Append a message to the chat history."""
        self.chat_history.append(role=role, msg=message)

    def embed_chat(self):
        """Generate and return the embeddings for the entire chat history."""
        if not self.chat_history.messages or len(self.chat_history) == 0:
            termcolor.cprint("No embeddings available: the chat history is empty.", "yellow")
            return None

        try:
            embedding, _ = self.chat_history.embed_chat()
            if embedding.size == 0:
                termcolor.cprint("No valid embeddings found.", "yellow")
                return None
            return embedding
        except ValueError as e:
            termcolor.cprint(f"Error generating embeddings: {e}", "red")
            return None

    def reset_history(self, system_prompt=True):
        """Reset the chat history."""
        self.chat_history.reset(system_prompt=system_prompt)

    def get_chat_history(self):
        """Return a list of all messages in the chat history."""
        return self.chat_history.to_list()

    def display_context_info(self):
        """Display the current context, including chat history, embeddings, and KV cache."""
        termcolor.cprint("Context Information:", "green")

        # Display chat history
        termcolor.cprint("Chat History:", "green")
        for i, msg in enumerate(self.get_chat_history()):
            role = msg.get('role', 'Unknown')
            message_content = msg.get('text', '[No message content]')  # Use 'text' instead of 'msg' based on how ChatHistory stores it
            termcolor.cprint(f"  {i+1}. {role.capitalize()}: {message_content}", "green")

        # Display embeddings
        embedding = self.embed_chat()
        if embedding is not None:
            termcolor.cprint(f"\nCurrent Embedding Shape: {embedding.shape}", "green")
            termcolor.cprint(f"First 5 Embedding Values (for context): {embedding[:, :5]}", "green")  # Example of displaying part of the embedding

        # Display KV Cache information
        kv_cache = self.chat_history.kv_cache
        if (kv_cache):
            termcolor.cprint(f"KV Cache contains {len(kv_cache)} tokens.", "green")
            termcolor.cprint(f"Raw KV Cache Weights (first 5): {kv_cache[:5]}", "yellow")  # Displaying part of KV cache for human readability
        else:
            termcolor.cprint("No KV Cache available.", "yellow")

    def display_functions(self):
        """Display available bot functions and their documentation."""
        termcolor.cprint("Available Functions:", "cyan")
        termcolor.cprint(BotFunctions.generate_docs(), "cyan")

    def process_command(self, command):
        """Process user commands to update settings, display information, or display functions."""
        if command.startswith("[max-tokens-in:"):
            try:
                value = int(command[len("[max-tokens-in:"):-1])
                self.set_max_input_tokens(value)
            except ValueError:
                print("Invalid max-tokens-in value.")
        elif command.startswith("[max-new-tokens:"):
            try:
                value = int(command[len("[max-new-tokens:"):-1])
                self.set_max_new_tokens(value)
            except ValueError:
                print("Invalid max-new-tokens value.")
        elif command.startswith("[temperature:"):
            try:
                value = float(command[len("[temperature:"):-1])
                self.set_temperature(value)
            except ValueError:
                print("Invalid temperature value.")
        elif command.startswith("[repetition-penalty:"):
            try:
                value = float(command[len("[repetition-penalty:"):-1])
                self.set_repetition_penalty(value)
            except ValueError:
                print("Invalid repetition-penalty value.")
        elif command.startswith("[top_p:"):
            try:
                value = float(command[len("[top_p:"):-1])
                self.set_top_p(value)
            except ValueError:
                print("Invalid top_p value.")
        elif command.startswith("[input-port:"):
            try:
                value = int(command[len("[input-port:"):-1])
                self.set_input_port(value)
            except ValueError:
                print("Invalid input-port value.")
        elif command.startswith("[output-port:"):
            try:
                value = int(command[len("[output-port:"):-1])
                self.set_output_port(value)
            except ValueError:
                print("Invalid output-port value.")
        elif command.startswith("[output-ip:"):
            try:
                value = command[len("[output-ip:"):-1]
                self.set_output_ip(value)
            except ValueError:
                print("Invalid output-ip value.")
        elif command.startswith("[system-prompt:"):
            if "+functions" in command:
                prompt_base = command[len("[system-prompt:"):-len("+functions]-")]
                self.modify_system_prompt(prompt_base, include_functions=True)
            else:
                prompt_base = command[len("[system-prompt:"):-1]
                self.modify_system_prompt(prompt_base)
        elif command == "[settings]":
            self.display_settings()
        elif command == "[context]":
            self.display_context_info()
        elif command == "[functions]":
            self.display_functions()
        else:
            print(f"Unknown command: {command}")

        self.save_settings()  # Save settings after processing any command

    def set_max_input_tokens(self, value):
        self.max_input_tokens = value
        termcolor.cprint(f"Settings updated: Max Input Tokens set to {value}", "blue")

    def set_max_new_tokens(self, value):
        self.max_new_tokens = value
        termcolor.cprint(f"Settings updated: Max New Tokens set to {value}", "blue")

    def set_temperature(self, value):
        self.temperature = value
        termcolor.cprint(f"Settings updated: Temperature set to {value}", "blue")

    def set_repetition_penalty(self, value):
        self.repetition_penalty = value
        termcolor.cprint(f"Settings updated: Repetition Penalty set to {value}", "blue")

    def set_top_p(self, value):
        self.top_p = value
        termcolor.cprint(f"Settings updated: Top-p set to {value}", "blue")

    def set_input_port(self, value):
        self.input_port = value
        termcolor.cprint(f"Settings updated: Input Port set to {value}", "blue")

    def set_output_port(self, value):
        self.output_port = value
        termcolor.cprint(f"Settings updated: Output Port set to {value}", "blue")

    def set_output_ip(self, value):
        self.output_ip = value
        termcolor.cprint(f"Settings updated: Output IP set to {value}", "blue")

    def modify_system_prompt(self, new_prompt, include_functions=False):
        """Modify the system prompt, optionally including function docs, and reset the chat history."""
        full_prompt = f"{new_prompt}"
        if include_functions:
            full_prompt += "\n\n" + BotFunctions.generate_docs()
        full_prompt += ""
        self.system_prompt = full_prompt
        self.reset_history(system_prompt=full_prompt)
        termcolor.cprint(f"Settings updated: System Prompt set to \"{full_prompt}\"", "magenta")

    def display_settings(self):
        """Display the current settings."""
        termcolor.cprint("General Settings:", "blue")
        termcolor.cprint(f"  Max Input Tokens     : {self.max_input_tokens}", "blue")
        termcolor.cprint(f"  Max New Tokens       : {self.max_new_tokens}", "blue")
        termcolor.cprint(f"  Temperature          : {self.temperature}", "blue")
        termcolor.cprint(f"  Repetition Penalty   : {self.repetition_penalty}", "blue")
        termcolor.cprint(f"  Top-p                : {self.top_p}", "blue")
        if self.input_port:
            termcolor.cprint(f"  Input Port           : {self.input_port}", "blue")
        else:
            termcolor.cprint(f"  Input Port           : Not set", "blue")
        if self.output_port:
            termcolor.cprint(f"  Output Port          : {self.output_port}", "blue")
        else:
            termcolor.cprint(f"  Output Port          : Not set", "blue")
        if self.output_ip:
            termcolor.cprint(f"  Output IP            : {self.output_ip}", "blue")
        else:
            termcolor.cprint(f"  Output IP            : Not set", "blue")
        termcolor.cprint(f"  System Prompt        : \"{self.system_prompt}\"", "magenta")

    async def generate_response(self, input_text):
        """Generate a response from the model based on the input text."""
        self.append_message('user', input_text)
        embedding = self.embed_chat()
        response = self.model.generate(
            embedding,
            streaming=True,
            functions=self.bot_functions,  # Provide access to functions
            kv_cache=self.chat_history.kv_cache,
            stop_tokens=self.chat_history.template.stop,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            top_p=self.top_p
        )

        writer = None
        if self.output_port and self.output_ip:
            try:
                reader, writer = await asyncio.open_connection(self.output_ip, self.output_port)
            except Exception as e:
                termcolor.cprint(f"Failed to connect to {self.output_ip}:{self.output_port}. Error: {e}", "red")
                return

        # Capture and send each token as it's generated
        for token in response:
            if token.strip() == "<|eot_id|>":
                continue
            print(token, end='', flush=True)
            bot_response = token

            # Append the model's response to the chat history
            self.append_message('bot', bot_response)

            # Send the token to the output port
            if writer:
                try:
                    writer.write(bot_response.encode())
                    await writer.drain()
                except Exception as e:
                    termcolor.cprint(f"Failed to send token to {self.output_ip}:{self.output_port}. Error: {e}", "red")
                    writer.close()
                    await writer.wait_closed()
                    writer = None  # Stop further attempts to send
        # Explicitly send an <eot> tag after the last token
        if writer:
            try:
                writer.write("<eot>".encode())
                await writer.drain()
            except Exception as e:
                termcolor.cprint(f"Failed to send <eot> to {self.output_ip}:{self.output_port}. Error: {e}", "red")

        print()

        if writer:
            writer.close()
            await writer.wait_closed()

    async def handle_input(self, reader, writer):
        """Handle incoming data from the input port."""
        buffer = ""

        try:
            while True:
                data = await reader.read(100)  # Adjust the buffer size as needed
                if not data:
                    break
                message = data.decode('utf-8').strip()

                # Check if the end-of-token marker is present
                if "<eot>" in message:
                    buffer += " " + message.replace("<eot>", "")  # Append message with a space before it, without the end-of-token marker
                    print()  # Add a newline before generating the response

                    # After all tokens are received, check if the buffer is a command
                    if buffer.strip().startswith("[") and buffer.strip().endswith("]"):
                        self.process_command(buffer.strip())
                        termcolor.cprint(f"Processed command: {buffer.strip()}", "green")
                    else:
                        await self.generate_response(buffer.strip())  # Generate and process response

                    buffer = ""  # Clear the buffer
                else:
                    buffer += " " + message  # Append the entire message to the buffer with a leading space
                    print(message, end=' ', flush=True)  # Print each message as it arrives

        except Exception as e:
            termcolor.cprint(f"Error in handling input: {e}", "red")
        finally:
            writer.close()
            await writer.wait_closed()



    async def handle_client(self, reader, writer):
        """Handles incoming connections and manages input processing."""
        termcolor.cprint("Client connected", "green")
        await self.handle_input(reader, writer)
        termcolor.cprint("Client disconnected", "yellow")
        # Do not close the connection; just reset the buffer and continue

    async def start_server(self):
        """Start the asyncio server to listen on the input port."""
        if not self.input_port:
            termcolor.cprint("Input port not set. Cannot start server.", "red")
            return

        server = await asyncio.start_server(self.handle_client, '127.0.0.1', self.input_port)
        addr = server.sockets[0].getsockname()
        termcolor.cprint(f"Serving on {addr}", "green")

        async with server:
            await server.serve_forever()

    async def terminal_input(self):
        """Handle input directly from the terminal if no input port is specified."""
        while True:
            message = await asyncio.get_event_loop().run_in_executor(None, input, "User: ")
            message = message.strip()
            if message.lower() == 'exit':
                break
            await self.generate_response(message)

    async def check_port(self):
        """Check if the specified input port has a listener and attempt to connect."""
        if self.input_port:
            try:
                reader, writer = await asyncio.open_connection('127.0.0.1', self.input_port)
                termcolor.cprint(f"Successfully connected to the listener on port {self.input_port}", "green")
                writer.close()
                await writer.wait_closed()
                return True
            except Exception as e:
                termcolor.cprint(f"No listener available on port {self.input_port}. Error: {e}", "red")
                return False
        return False

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2", help="Model to use for the LLM.")
    parser.add_argument("--max-context-len", type=int, default=1024, help="Maximum context length for the model.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--prompt", type=str, action='append', default=[""], help="Prompts for the LLM.")
    parser.add_argument("--api", type=str, default="mlc", help="API to use for the LLM.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for the LLM's response generation.")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty for the LLM's response generation.")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p (nucleus) sampling for the LLM's response generation.")
    parser.add_argument("--disable-plugins", action='store_true', help="Disable plugins for the LLM.")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Minimum number of new tokens to generate.")
    parser.add_argument("--do-sample", action='store_true', help="Enable sampling for the LLM's response generation.")
    parser.add_argument("--wrap-tokens", type=int, default=None, help="Wrap tokens to keep the most recent tokens.")
    parser.add_argument("--input-port", type=int, help="Port to listen for input.")
    parser.add_argument("--output-port", type=int, help="Port to send output.")
    parser.add_argument("--output-ip", type=str, default="127.0.0.1", help="IP to send output to.")
    return parser.parse_args()

def print_available_commands():
    print("You can update the following settings during your session:")
    print("  - [max-tokens-in:<value>]       : Set max input tokens")
    print("  - [max-new-tokens:<value>]      : Set max new tokens")
    print("  - [temperature:<value>]         : Set temperature")
    print("  - [repetition-penalty:<value>]  : Set repetition penalty")
    print("  - [top_p:<value>]               : Set top-p (nucleus) sampling")
    print("  - [input-port:<value>]          : Set the input port")
    print("  - [output-port:<value>]         : Set the output port")
    print("  - [output-ip:<value>]           : Set the output IP")
    print("  - [system-prompt:<text>]        : Set the system prompt")
    print("  - [system-prompt:<text>+functions] : Set the system prompt and include function docs")
    print("  - [settings]                    : Display current settings")
    print("  - [context]                     : Display current context embedding and kv cache")
    print("  - [functions]                   : Display available functions and their documentation")
    print()

async def main_loop(chat_manager):
    while True:
        user_input = input("User: ").strip().lower()

        if user_input == 'exit':
            break

        if any(user_input.startswith(prefix) for prefix in [
            "[max-tokens-in:", "[max-new-tokens:", "[temperature:", 
            "[repetition-penalty:", "[top_p:", "[system-prompt:", 
            "[settings]", "[context]", "[functions]", 
            "[input-port:", "[output-port:", "[output-ip:"
        ]):
            chat_manager.process_command(user_input)
            continue

        await chat_manager.generate_response(user_input)

        # Print a single newline to separate responses
        print()

async def main():
    args = parse_args()

    global model
    model = NanoLLM.from_pretrained(
        model=args.model,
        api=args.api,
        quantization='q4f16_ft',
        max_context_len=args.max_context_len,
    )
    
    chat_manager = ChatManager(
        model=model,
        system_prompt="You are a helpful and friendly AI assistant.",
        max_input_tokens=args.max_context_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        top_p=args.top_p,
        input_port=args.input_port,
        output_port=args.output_port,
        output_ip=args.output_ip
    )

    # Start the server for the input port immediately
    if args.input_port:
        await chat_manager.start_server()  # Ensure the server starts before anything else

    print_available_commands()

    # Run the main loop for terminal input
    await main_loop(chat_manager)

if __name__ == '__main__':
    asyncio.run(main())
