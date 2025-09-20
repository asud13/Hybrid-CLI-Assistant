
# Hybrid CLI Assistant

A privacy-focused command-line AI assistant that combines local LLMs with a cloud-based GPT-4 fallback. Designed for speed, reliability, and security, the assistant prioritizes local execution whenever possible, with seamless failover to GPT-4 when needed.



## Features
- Privacy-first: no stored data.
- Hybrid model execution: runs queries on local LLMs (via Ollama CLI or llama.cpp) with GPT-4 fallback.
- Responsive interaction: asynchronous querying with timeouts for robust handling.
- Graceful fallback: if local models hang or fail, the system automatically switches to GPT-4.
- Terminal UI enhancements for clear, stable, and user-friendly CLI.


## Installation

Clone my project

```bash
  git clone https://github.com/asud13/Hybrid-CLI-Assistant.git
  cd hybrid-cli-assistant
```

Set up python environment 

```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
pip install -r requirements.txt
```

Install local model dependencies:

[![Ollama CLI](https://img.shields.io/badge/Ollama_CLI-000?style=for-the-badge&logo=python&logoColor=white)](https://ollama.ai/download)  
[![llama.cpp](https://img.shields.io/badge/llama.cpp-000?style=for-the-badge&logo=ollama&logoColor=white)](https://github.com/ggerganov/llama.cpp)

Configure API keys for GPT-4 fallback:

(BYOK) -> Bring your own key! 

```bash
export OPENAI_API_KEY="your_api_key_here"
```


## Usage

Run the assistant:

```bash
  python launchassistant.py
```

The assistant will attempt to use local LLMs first. If a timeout or failure occurs, it will automatically fall back to GPT-4.

Configuration

You can adjust:

- Timeouts for local model responses.

- Model preference order (local-first vs. cloud-first).

- Logging verbosity for debugging.

## Project Structure
```bash
  hybrid-cli-assistant/
  │── launchassistant.py     # Entry point for launching the assistant
  │── simple_hybrid_cli.py   # Core hybrid CLI logic
  │── requirements.txt       # Python dependencies
  │── README.md              # Project documentation

```


## License

[MIT](https://choosealicense.com/licenses/mit/)

