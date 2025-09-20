#!/usr/bin/env python3
"""
Hybrid CLI Assistant
A command-line AI assistant that combines local LLMs with cloud-based GPT-4 fallback.
Emphasizes privacy, security, and robust performance.
"""

import asyncio
import subprocess
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Optional, Dict, Any
import signal
import shutil
from pathlib import Path

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: OpenAI library not installed. Cloud fallback will be unavailable.")

try:
    import rich
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.spinner import Spinner
    from rich.live import Live
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Warning: Rich library not installed. Enhanced UI will be unavailable.")


class Config:
    """Configuration management for the CLI assistant."""
    
    def __init__(self):
        self.config_dir = Path.home() / ".hybrid-cli-assistant"
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            "openai_api_key": "",
            "local_model_timeout": 30,
            "local_model_command": "ollama run llama2",
            "fallback_model": "gpt-4",
            "max_tokens": 1000,
            "temperature": 0.7,
            "verbose": False
        }
        self.load_config()
    
    def load_config(self):
        """Load configuration from file or create default."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    self.config = {**self.default_config, **json.load(f)}
            except (json.JSONDecodeError, IOError):
                self.config = self.default_config.copy()
                self.save_config()
        else:
            self.config = self.default_config.copy()
            self.save_config()
    
    def save_config(self):
        """Save current configuration to file."""
        self.config_dir.mkdir(exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str) -> Any:
        """Get configuration value."""
        return self.config.get(key)
    
    def set(self, key: str, value: Any):
        """Set configuration value and save."""
        self.config[key] = value
        self.save_config()


class TerminalUI:
    """Enhanced terminal UI with Rich library support."""
    
    def __init__(self):
        self.console = Console() if RICH_AVAILABLE else None
    
    def print_info(self, message: str):
        """Print informational message."""
        if self.console:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            print(f"INFO: {message}")
    
    def print_error(self, message: str):
        """Print error message."""
        if self.console:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            print(f"ERROR: {message}")
    
    def print_success(self, message: str):
        """Print success message."""
        if self.console:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            print(f"SUCCESS: {message}")
    
    def print_warning(self, message: str):
        """Print warning message."""
        if self.console:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            print(f"WARNING: {message}")
    
    def print_response(self, response: str, source: str = ""):
        """Print AI response with formatting."""
        if self.console:
            if source:
                header = f"Response from {source}"
            else:
                header = "Response"
            
            panel = Panel(
                Markdown(response),
                title=header,
                border_style="cyan",
                padding=(1, 2)
            )
            self.console.print(panel)
        else:
            if source:
                print(f"\n--- Response from {source} ---")
            else:
                print(f"\n--- Response ---")
            print(response)
            print("---")
    
    def show_spinner(self, message: str):
        """Show a spinner with message."""
        if self.console:
            return self.console.status(f"[cyan]{message}...", spinner="dots")
        else:
            print(f"{message}...")
            return None


class LocalLLMInterface:
    """Interface for local LLM models."""
    
    def __init__(self, config: Config, ui: TerminalUI):
        self.config = config
        self.ui = ui
        self.process = None
    
    async def query_ollama(self, prompt: str) -> Optional[str]:
        """Query Ollama model asynchronously."""
        try:
            cmd = self.config.get("local_model_command")
            
            # Handle Windows paths with spaces
            if sys.platform == "win32" and cmd.startswith('C:'):
                # Split on .exe to handle paths with spaces
                if '.exe' in cmd:
                    exe_index = cmd.find('.exe') + 4
                    executable = cmd[:exe_index].strip('"')
                    args = cmd[exe_index:].strip().split() if exe_index < len(cmd) else []
                    full_cmd = [executable] + args
                else:
                    full_cmd = cmd.split()
            else:
                full_cmd = cmd.split()
            
            # Create a temporary file for the prompt to avoid issues with input
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(prompt + '\n/bye\n')
                temp_file = f.name
            
            try:
                # Use shell=True on Windows for better compatibility
                if sys.platform == "win32":
                    process = await asyncio.create_subprocess_shell(
                        f'"{full_cmd[0]}" {" ".join(full_cmd[1:])} < "{temp_file}"',
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        text=True
                    )
                else:
                    process = await asyncio.create_subprocess_exec(
                        *full_cmd,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                        text=True
                    )
                
                if sys.platform == "win32":
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.config.get("local_model_timeout")
                    )
                else:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input=prompt + '\n/bye\n'),
                        timeout=self.config.get("local_model_timeout")
                    )
                
                if process.returncode == 0 and stdout:
                    # Clean up the output - remove the prompt echo and bye message
                    lines = stdout.strip().split('\n')
                    # Filter out empty lines and command echoes
                    response_lines = [line for line in lines if line.strip() and not line.strip().startswith('>') and '/bye' not in line.lower()]
                    if response_lines:
                        return '\n'.join(response_lines).strip()
                    return None
                else:
                    if stderr:
                        self.ui.print_error(f"Local model error: {stderr}")
                    return None
                    
            except asyncio.TimeoutError:
                self.ui.print_warning("Local model timed out")
                if process:
                    try:
                        process.kill()
                        await process.wait()
                    except:
                        pass
                return None
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file)
                except:
                    pass
                
        except Exception as e:
            self.ui.print_error(f"Failed to query local model: {e}")
            return None
    
    async def query_llama_cpp(self, prompt: str) -> Optional[str]:
        """Query llama.cpp model asynchronously."""
        try:
            # Check if llama.cpp is available
            if not shutil.which("llama"):
                return None
            
            cmd = [
                "llama",
                "-p", prompt,
                "-n", str(self.config.get("max_tokens")),
                "--temp", str(self.config.get("temperature")),
                "-b", "1"  # Batch size
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.config.get("local_model_timeout")
                )
                
                if process.returncode == 0:
                    return stdout.strip()
                else:
                    return None
                    
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                return None
                
        except Exception:
            return None
    
    async def query_local_models(self, prompt: str) -> Optional[tuple]:
        """Try multiple local model interfaces."""
        # Try Ollama first
        result = await self.query_ollama(prompt)
        if result:
            return result, "Ollama"
        
        # Try llama.cpp as fallback
        result = await self.query_llama_cpp(prompt)
        if result:
            return result, "llama.cpp"
        
        return None


class CloudLLMInterface:
    """Interface for cloud-based LLM models."""
    
    def __init__(self, config: Config, ui: TerminalUI):
        self.config = config
        self.ui = ui
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize OpenAI client if available."""
        if not OPENAI_AVAILABLE:
            return
        
        api_key = self.config.get("openai_api_key")
        if api_key:
            try:
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                self.ui.print_error(f"Failed to initialize OpenAI client: {e}")
    
    async def query_openai(self, prompt: str) -> Optional[str]:
        """Query OpenAI API asynchronously."""
        if not self.client:
            return None
        
        try:
            # Run the synchronous OpenAI call in a thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.config.get("fallback_model"),
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.config.get("max_tokens"),
                    temperature=self.config.get("temperature")
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.ui.print_error(f"OpenAI API error: {e}")
            return None


class HybridCLIAssistant:
    """Main hybrid CLI assistant class."""
    
    def __init__(self):
        self.config = Config()
        self.ui = TerminalUI()
        self.local_llm = LocalLLMInterface(self.config, self.ui)
        self.cloud_llm = CloudLLMInterface(self.config, self.ui)
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.ui.print_info("Shutting down gracefully...")
        self.running = False
        sys.exit(0)
    
    async def process_query(self, prompt: str) -> bool:
        """Process a user query with local-first approach and cloud fallback."""
        if not prompt.strip():
            return True
        
        # Try local models first
        spinner = self.ui.show_spinner("Querying local models")
        
        try:
            if spinner and hasattr(spinner, '__enter__'):
                with spinner:
                    local_result = await self.local_llm.query_local_models(prompt)
            else:
                local_result = await self.local_llm.query_local_models(prompt)
            
            if local_result:
                response, source = local_result
                self.ui.print_response(response, f"Local ({source})")
                return True
            
        except Exception as e:
            if self.config.get("verbose"):
                self.ui.print_error(f"Local model query failed: {e}")
        
        # Fallback to cloud API
        self.ui.print_warning("Local models unavailable, trying cloud fallback...")
        
        spinner = self.ui.show_spinner("Querying cloud API")
        
        try:
            if spinner and hasattr(spinner, '__enter__'):
                with spinner:
                    cloud_result = await self.cloud_llm.query_openai(prompt)
            else:
                cloud_result = await self.cloud_llm.query_openai(prompt)
            
            if cloud_result:
                self.ui.print_response(cloud_result, "OpenAI GPT-4")
                return True
            else:
                self.ui.print_error("All AI services are currently unavailable")
                return True
                
        except Exception as e:
            self.ui.print_error(f"Cloud API query failed: {e}")
            return True
    
    def configure(self):
        """Interactive configuration setup."""
        self.ui.print_info("Configuration Setup")
        
        # OpenAI API Key
        current_key = self.config.get("openai_api_key")
        if current_key:
            key_display = current_key[:8] + "..." + current_key[-4:] if len(current_key) > 12 else "Set"
        else:
            key_display = "Not set"
        
        new_key = input(f"OpenAI API Key ({key_display}): ").strip()
        if new_key:
            self.config.set("openai_api_key", new_key)
        
        # Local model command
        current_cmd = self.config.get("local_model_command")
        new_cmd = input(f"Local model command ({current_cmd}): ").strip()
        if new_cmd:
            self.config.set("local_model_command", new_cmd)
        
        # Timeout
        current_timeout = self.config.get("local_model_timeout")
        timeout_input = input(f"Local model timeout in seconds ({current_timeout}): ").strip()
        if timeout_input.isdigit():
            self.config.set("local_model_timeout", int(timeout_input))
        
        self.ui.print_success("Configuration saved!")
    
    def show_status(self):
        """Show system status and available models."""
        self.ui.print_info("System Status:")
        
        # Check Ollama
        ollama_available = shutil.which("ollama") is not None
        self.ui.print_success("Ollama: Available") if ollama_available else self.ui.print_error("Ollama: Not found")
        
        # Check llama.cpp
        llama_cpp_available = shutil.which("llama") is not None
        self.ui.print_success("llama.cpp: Available") if llama_cpp_available else self.ui.print_error("llama.cpp: Not found")
        
        # Check OpenAI
        openai_configured = bool(self.config.get("openai_api_key")) and OPENAI_AVAILABLE
        self.ui.print_success("OpenAI API: Configured") if openai_configured else self.ui.print_error("OpenAI API: Not configured")
        
        print(f"\nTimeout: {self.config.get('local_model_timeout')}s")
        print(f"Local command: {self.config.get('local_model_command')}")
        print(f"Fallback model: {self.config.get('fallback_model')}")
    
    async def interactive_mode(self):
        """Run in interactive mode."""
        self.ui.print_info("Hybrid CLI Assistant - Interactive Mode")
        self.ui.print_info("Type 'exit', 'quit', or Ctrl+C to exit")
        
        while self.running:
            try:
                prompt = input("\n🤖 You: ").strip()
                
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                if prompt.lower() == 'status':
                    self.show_status()
                    continue
                
                if prompt.lower() == 'config':
                    self.configure()
                    continue
                
                await self.process_query(prompt)
                
            except EOFError:
                break
            except KeyboardInterrupt:
                break
        
        self.ui.print_info("Goodbye!")
    
    async def single_query_mode(self, query: str):
        """Process a single query and exit."""
        await self.process_query(query)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hybrid CLI Assistant")
    parser.add_argument("query", nargs="*", help="Query to process (if not provided, enters interactive mode)")
    parser.add_argument("--config", action="store_true", help="Configure the assistant")
    parser.add_argument("--status", action="store_true", help="Show system status")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    assistant = HybridCLIAssistant()
    
    if args.verbose:
        assistant.config.set("verbose", True)
    
    if args.config:
        assistant.configure()
        return
    
    if args.status:
        assistant.show_status()
        return
    
    try:
        if args.query:
            # Single query mode
            query = " ".join(args.query)
            asyncio.run(assistant.single_query_mode(query))
        else:
            # Interactive mode
            asyncio.run(assistant.interactive_mode())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        assistant.ui.print_error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()