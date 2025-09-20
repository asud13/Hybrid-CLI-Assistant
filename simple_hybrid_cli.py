#!/usr/bin/env python3
"""
Simple Hybrid CLI Assistant
A simplified version focusing on core functionality
"""

import asyncio
import subprocess
import sys
import json
import argparse
import textwrap
import time
from datetime import datetime
from pathlib import Path

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not installed. HTTP API unavailable.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Cloud fallback unavailable.")


class SimpleConfig:
    """Simple configuration management."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".simple-hybrid-cli"
        self.config_file = self.config_dir / "config.json"
        self.config = {
            "openai_api_key": "",
            "ollama_model": "llama2",
            "timeout": 30,
            "max_tokens": 500,
            "temperature": 0.7,
            "show_model_source": True,
            "show_timestamps": False,
            "show_query_numbers": True,
            "wrap_width": 80,
            "use_colors": True,
            "save_conversations": False,
            "conversation_file": "conversations.json"
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
            except:
                pass
    
    def save_config(self):
        """Save configuration."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key):
        return self.config.get(key)
    
    def set(self, key, value):
        self.config[key] = value
        self.save_config()
    
    def get_all(self):
        """Get all configuration as a dictionary."""
        return self.config.copy()


class OutputFormatter:
    """Handle output formatting and styling."""
    
    def __init__(self, config):
        self.config = config
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'gray': '\033[90m'
        } if self.config.get('use_colors') else {k: '' for k in ['reset', 'bold', 'dim', 'red', 'green', 'yellow', 'blue', 'magenta', 'cyan', 'white', 'gray']}
    
    def colorize(self, text, color):
        """Add color to text."""
        if not self.config.get('use_colors'):
            return text
        return f"{self.colors.get(color, '')}{text}{self.colors['reset']}"
    
    def wrap_text(self, text, width=None):
        """Wrap text to specified width."""
        if width is None:
            width = self.config.get('wrap_width', 80)
        
        if width <= 0:
            return text
        
        # Handle multiple paragraphs
        paragraphs = text.split('\n\n')
        wrapped_paragraphs = []
        
        for paragraph in paragraphs:
            if paragraph.strip():
                wrapped = textwrap.fill(paragraph.strip(), width=width)
                wrapped_paragraphs.append(wrapped)
        
        return '\n\n'.join(wrapped_paragraphs)
    
    def format_timestamp(self):
        """Get formatted timestamp if enabled."""
        if not self.config.get('show_timestamps'):
            return ""
        return f"{self.colorize(datetime.now().strftime('%H:%M:%S'), 'dim')} "
    
    def print_header(self, title, char="=", color="cyan"):
        """Print a formatted header."""
        width = self.config.get('wrap_width', 80)
        header_line = char * min(width, len(title) + 10)
        centered_title = title.center(len(header_line))
        
        print(self.colorize(header_line, color))
        print(self.colorize(centered_title, 'bold'))
        print(self.colorize(header_line, color))
    
    def print_response(self, response, source, query_num=None):
        """Format and print AI response."""
        timestamp = self.format_timestamp()
        
        # Header with source info
        if query_num and self.config.get('show_query_numbers'):
            header = f"🤖 Response #{query_num}"
        else:
            header = "🤖 Assistant"
        
        if self.config.get('show_model_source'):
            header += f" ({source})"
        
        # Print header with timestamp
        print(f"\n{timestamp}{self.colorize(header, 'green')}")
        
        # Format and wrap response
        wrapped_response = self.wrap_text(response)
        
        # Print response with nice formatting
        print(self.colorize("┌" + "─" * (self.config.get('wrap_width', 80) - 2) + "┐", 'dim'))
        
        for line in wrapped_response.split('\n'):
            if line.strip():
                print(f"{self.colorize('│', 'dim')} {line}")
            else:
                print(f"{self.colorize('│', 'dim')}")
        
        print(self.colorize("└" + "─" * (self.config.get('wrap_width', 80) - 2) + "┘", 'dim'))
    
    def print_status(self, message, status_type="info"):
        """Print status message with formatting."""
        timestamp = self.format_timestamp()
        icons = {
            'info': '🔄',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'thinking': '🧠'
        }
        colors = {
            'info': 'blue',
            'success': 'green',
            'warning': 'yellow',
            'error': 'red',
            'thinking': 'magenta'
        }
        
        icon = icons.get(status_type, '🔄')
        color = colors.get(status_type, 'white')
        
        print(f"{timestamp}{icon} {self.colorize(message, color)}")
    
    def print_error(self, message):
        """Print error message."""
        self.print_status(f"Error: {message}", "error")
    
    def print_info(self, message):
        """Print info message."""
        self.print_status(message, "info")
    
    def print_success(self, message):
        """Print success message."""
        self.print_status(message, "success")
    
    def print_thinking(self, message="Thinking..."):
        """Print thinking status."""
        self.print_status(message, "thinking")
class SimpleAssistant:
    """Simple hybrid assistant."""
    
    def __init__(self):
        self.config = SimpleConfig()
        self.formatter = OutputFormatter(self.config)
        self.openai_client = None
        self.conversation_history = []
        self._setup_openai()
    
    def _setup_openai(self):
        """Setup OpenAI client if available."""
        if OPENAI_AVAILABLE and self.config.get("openai_api_key"):
            try:
                self.openai_client = openai.OpenAI(api_key=self.config.get("openai_api_key"))
            except Exception as e:
                self.formatter.print_error(f"OpenAI setup failed: {e}")
    
    def query_ollama_http(self, prompt):
        """Try Ollama HTTP API."""
        if not REQUESTS_AVAILABLE:
            return None
        
        try:
            url = "http://localhost:11434/api/generate"
            data = {
                "model": self.config.get("ollama_model"),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": self.config.get("max_tokens"),
                    "temperature": self.config.get("temperature")
                }
            }
            
            response = requests.post(url, json=data, timeout=self.config.get("timeout"))
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
        except Exception as e:
            if self.config.get("verbose", False):
                self.formatter.print_error(f"HTTP API error: {e}")
        
        return None
    
    def query_ollama_cli(self, prompt):
        """Try Ollama CLI."""
        try:
            # Find Ollama executable
            ollama_path = "C:\\Users\\anisu\\AppData\\Local\\Programs\\Ollama\\ollama.exe"
            
            # Simple command approach
            cmd = f'echo "{prompt}" | "{ollama_path}" run {self.config.get("ollama_model")}'
            
            result = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.get("timeout")
            )
            
            if result.returncode == 0 and result.stdout:
                output = result.stdout.strip()
                if output and output != prompt:
                    return output
            
            if result.stderr and self.config.get("verbose", False):
                self.formatter.print_error(f"CLI error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            self.formatter.print_error("Ollama CLI timed out")
        except Exception as e:
            if self.config.get("verbose", False):
                self.formatter.print_error(f"CLI query failed: {e}")
        
        return None
    
    def query_openai(self, prompt):
        """Try OpenAI API."""
        if not self.openai_client:
            return None
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.get("max_tokens"),
                temperature=self.config.get("temperature")
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.formatter.print_error(f"OpenAI error: {e}")
        
        return None
    
    def process_query(self, prompt):
        """Process a query with local-first approach."""
        if not prompt.strip():
            return
        
        print("🔄 Trying local models...")
        
        # Try HTTP API first
        response = self.query_ollama_http(prompt)
        if response:
            print(f"\n✅ Response from Ollama (HTTP):")
            print(f"{response}\n")
            return
        
        # Try CLI approach
        print("🔄 Trying CLI approach...")
        response = self.query_ollama_cli(prompt)
        if response:
            print(f"\n✅ Response from Ollama (CLI):")
            print(f"{response}\n")
            return
        
        # Fallback to OpenAI
        print("⚠️  Local models failed, trying OpenAI...")
        response = self.query_openai(prompt)
        if response:
            print(f"\n✅ Response from OpenAI:")
            print(f"{response}\n")
            return
        
        print("❌ All AI services are unavailable")
    
    def configure(self):
        """Enhanced configuration setup."""
        self.formatter.print_header("🔧 Configuration Setup", color="cyan")
        
        config_options = [
            {
                "key": "ollama_model",
                "prompt": "Ollama model name",
                "current": self.config.get("ollama_model"),
                "type": "string"
            },
            {
                "key": "timeout",
                "prompt": "Request timeout (seconds)",
                "current": self.config.get("timeout"),
                "type": "int"
            },
            {
                "key": "max_tokens",
                "prompt": "Maximum response tokens",
                "current": self.config.get("max_tokens"),
                "type": "int"
            },
            {
                "key": "temperature",
                "prompt": "Response creativity (0.0-2.0)",
                "current": self.config.get("temperature"),
                "type": "float"
            },
            {
                "key": "wrap_width",
                "prompt": "Text wrap width (0 for no wrap)",
                "current": self.config.get("wrap_width"),
                "type": "int"
            },
            {
                "key": "show_model_source",
                "prompt": "Show which model responded",
                "current": self.config.get("show_model_source"),
                "type": "bool"
            },
            {
                "key": "show_timestamps",
                "prompt": "Show timestamps",
                "current": self.config.get("show_timestamps"),
                "type": "bool"
            },
            {
                "key": "show_query_numbers",
                "prompt": "Show query numbers",
                "current": self.config.get("show_query_numbers"),
                "type": "bool"
            },
            {
                "key": "use_colors",
                "prompt": "Use colored output",
                "current": self.config.get("use_colors"),
                "type": "bool"
            },
            {
                "key": "save_conversations",
                "prompt": "Save conversation history",
                "current": self.config.get("save_conversations"),
                "type": "bool"
            }
        ]
        
        print(f"\n{self.formatter.colorize('Current Configuration:', 'bold')}")
        print(f"{self.formatter.colorize('=' * 40, 'dim')}")
        
        changes_made = False
        
        for option in config_options:
            key = option["key"]
            prompt_text = option["prompt"]
            current = option["current"]
            value_type = option["type"]
            
            # Display current value
            if value_type == "bool":
                display_value = "Yes" if current else "No"
            else:
                display_value = str(current)
            
            user_input = input(f"{prompt_text} ({self.formatter.colorize(display_value, 'cyan')}): ").strip()
            
            if user_input:
                try:
                    if value_type == "int":
                        new_value = int(user_input)
                    elif value_type == "float":
                        new_value = float(user_input)
                    elif value_type == "bool":
                        new_value = user_input.lower() in ['yes', 'y', 'true', '1', 'on']
                    else:
                        new_value = user_input
                    
                    if new_value != current:
                        self.config.set(key, new_value)
                        changes_made = True
                        self.formatter.print_success(f"Updated {prompt_text.lower()}")
                        
                except ValueError:
                    self.formatter.print_error(f"Invalid value for {prompt_text.lower()}")
        
        # OpenAI API Key (special handling)
        current_key = self.config.get("openai_api_key")
        key_display = "Set" if current_key else "Not set"
        new_key = input(f"\nOpenAI API Key ({self.formatter.colorize(key_display, 'cyan')}): ").strip()
        if new_key:
            self.config.set("openai_api_key", new_key)
            self._setup_openai()
            changes_made = True
            self.formatter.print_success("Updated OpenAI API key")
        
        if changes_made:
            # Refresh formatter with new config
            self.formatter = OutputFormatter(self.config)
            self.formatter.print_success("Configuration saved successfully!")
        else:
            self.formatter.print_info("No changes made")
        
        # Show final configuration
        print(f"\n{self.formatter.colorize('Final Configuration:', 'bold')}")
        self._show_config_summary()
    
    def _show_config_summary(self):
        """Show a summary of current configuration."""
        config = self.config.get_all()
        
        categories = {
            "Model Settings": ["ollama_model", "timeout", "max_tokens", "temperature"],
            "Display Options": ["wrap_width", "use_colors", "show_model_source", "show_timestamps", "show_query_numbers"],
            "Storage Options": ["save_conversations", "conversation_file"],
            "API Keys": ["openai_api_key"]
        }
        
        for category, keys in categories.items():
            print(f"\n{self.formatter.colorize(category + ':', 'yellow')}")
            for key in keys:
                if key in config:
                    value = config[key]
                    if key == "openai_api_key":
                        display_value = "Set" if value else "Not set"
                    elif isinstance(value, bool):
                        display_value = "Yes" if value else "No"
                    else:
                        display_value = str(value)
                    
                    print(f"  {key.replace('_', ' ').title()}: {self.formatter.colorize(display_value, 'cyan')}")
    
    def show_status(self):
        """Enhanced system status display."""
        self.formatter.print_header("🔍 System Status", color="blue")
        
        # Check if Ollama HTTP API is responding
        if REQUESTS_AVAILABLE:
            try:
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                if response.status_code == 200:
                    self.formatter.print_success("Ollama HTTP API: Available")
                    models = response.json().get("models", [])
                    if models:
                        model_names = [m['name'] for m in models]
                        print(f"   Available models: {self.formatter.colorize(', '.join(model_names), 'cyan')}")
                        
                        # Check if current model exists
                        current_model = self.config.get('ollama_model')
                        if current_model + ':latest' in model_names or current_model in model_names:
                            self.formatter.print_success(f"Current model '{current_model}' is available")
                        else:
                            self.formatter.print_error(f"Current model '{current_model}' not found!")
                else:
                    self.formatter.print_error("Ollama HTTP API: Not responding")
            except Exception as e:
                self.formatter.print_error(f"Ollama HTTP API: Not available ({e})")
        else:
            self.formatter.print_error("Ollama HTTP API: requests library not installed")
        
        # Check OpenAI
        if self.openai_client:
            self.formatter.print_success("OpenAI API: Configured")
        else:
            self.formatter.print_error("OpenAI API: Not configured")
        
        # Show current configuration
        print()
        self._show_config_summary()
        
        # Show conversation history status
        if self.config.get("save_conversations"):
            history_file = Path.home() / ".simple-hybrid-cli" / self.config.get("conversation_file")
            if history_file.exists():
                try:
                    with open(history_file, 'r') as f:
                        history = json.load(f)
                    self.formatter.print_info(f"Conversation history: {len(history)} entries saved")
                except:
                    self.formatter.print_error("Conversation history: File exists but couldn't be read")
            else:
                self.formatter.print_info("Conversation history: No saved conversations yet")

    
    def interactive_mode(self):
        """Interactive mode."""
        print("=" * 50)
        print("🤖 Simple Hybrid CLI Assistant")
        print("=" * 50)
        print("Commands:")
        print("  'exit' or 'quit' - Exit the assistant")
        print("  'status' - Show system status")
        print("  'config' - Configure settings")
        print("  'help' - Show this help")
        print("  'clear' - Clear the screen")
        print("=" * 50)
        
        conversation_count = 0
        
        while True:
            try:
                prompt = input(f"\n💬 You: ").strip()
                
                if prompt.lower() in ['exit', 'quit', 'bye']:
                    print("👋 Thanks for using the assistant!")
                    break
                elif prompt.lower() == 'status':
                    print()
                    self.show_status()
                elif prompt.lower() == 'config':
                    print()
                    self.configure()
                elif prompt.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  'exit', 'quit', 'bye' - Exit")
                    print("  'status' - System status")
                    print("  'config' - Configure settings")
                    print("  'clear' - Clear screen")
                    print("  Or just type any question!")
                elif prompt.lower() == 'clear':
                    import os
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("🤖 Simple Hybrid CLI Assistant")
                    print("Type 'help' for commands")
                elif prompt:
                    conversation_count += 1
                    print(f"\n[Query #{conversation_count}]")
                    self.process_query(prompt)
                    
            except (EOFError, KeyboardInterrupt):
                print("\n\n👋 Goodbye!")
                break
        
        print(f"Session ended after {conversation_count} queries.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Simple Hybrid CLI Assistant")
    parser.add_argument("query", nargs="*", help="Query to process")
    parser.add_argument("--config", action="store_true", help="Configure")
    parser.add_argument("--status", action="store_true", help="Show status")
    
    args = parser.parse_args()
    assistant = SimpleAssistant()
    
    if args.config:
        assistant.configure()
    elif args.status:
        assistant.show_status()
    elif args.query:
        assistant.process_query(" ".join(args.query))
    else:
        assistant.interactive_mode()


if __name__ == "__main__":
    main()