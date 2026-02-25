# Audio Recognition and Summarization Project

This project provides a complete pipeline for audio processing that includes speaker diarization, speech-to-text transcription, and conversation summarization using LLMs.

## Features

- Speaker diarization to identify who spoke when
- High-quality speech-to-text transcription using Whisper
- Text formatting for readability
- Conversation summarization using LLMs (Ollama or Groq API)
- Support for multiple languages (default Russian)

## Architecture

The project follows a modular architecture with the following components:

1. **Audio Processing**: Handles speaker diarization and transcription
2. **Text Formatting**: Prepares transcribed text for LLM processing
3. **Summarization**: Generates conversation summaries using LLMs
4. **Configuration**: Manages application settings

## Prerequisites

- Python 3.9+
- FFmpeg installed in the system
- Hugging Face account and access token for pyannote.audio models
- (Optional) Groq API key or local Ollama installation

## Installation

1. Clone the repository
2. Install system dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install ffmpeg
   
   # macOS
   brew install ffmpeg
   ```

3. Create a virtual environment and install Python dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```



## Configuration

Set the required environment variables:

```bash

export AUDIO_FILE="path_to_your_audio_file.mp3"
```

For LLM selection:
моделью ollama запущенной http://192.168.23.10:11434
## Usage

### Command Line Usage

Run the main script:

```bash
python -m src.main
```

The application will:
1. Process the audio file with speaker diarization
2. Transcribe the speech to text
3. Format the transcript with speaker labels
4. Generate a summary using the selected LLM
5. Save the full transcript to `transcript.txt`

### Web Interface Usage

Start the Flask web server:

```bash
python -m src.api.app
```

Then navigate to `http://localhost:5000` in your browser to access the web interface. You can upload an audio file through the interface and get the results displayed in your browser.

## Docker Deployment

### CLI Version

Build and run the CLI version with Docker:

```bash
docker build -t audio-summarizer .
docker run -e HF_TOKEN="your_token" -e AUDIO_FILE="input.mp3" -v $(pwd):/app audio-summarizer
```

### Web Interface Version

To run the web interface in Docker, build the image as above and then run:

```bash
docker run -e HF_TOKEN="your_token" -p 5000:5000 audio-summarizer python -m src.api.app
```

Then access the web interface at `http://localhost:5000`.

## Project Structure

```
/workspace/
├── src/
│   ├── main.py                 # Main entry point
│   ├── models/                 # Data models
│   ├── services/               # Business logic services
│   │   ├── audio_processor.py  # Audio processing service
│   │   ├── text_formatter.py   # Text formatting service
│   │   └── summarizer.py       # Summarization service
│   ├── utils/                  # Utility functions
│   │   └── config.py           # Configuration management
│   └── api/                    # API endpoints (future)
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Dockerfile                 # Docker configuration
└── README.md
```

## Future Enhancements

1. **Speaker Identification**: Add functionality to assign names to speakers (SPEAKER_00 → John)
2. **Search Capability**: Store results in a vector database for semantic search
3. **Web Interface**: Create a Streamlit or Flask interface for file uploads
4. **Batch Processing**: Support processing multiple files
5. **Output Formats**: Export summaries in different formats (JSON, PDF, etc.)

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
