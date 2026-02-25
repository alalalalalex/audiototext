"""
Flask web application for the audio recognition and summarization project.
"""
import os
import tempfile
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

from src.services.audio_processor import AudioProcessor
from src.services.text_formatter import TextFormatter
from src.services.summarizer import Summarizer
from src.utils.config import Config

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize services
config = Config()
audio_processor = AudioProcessor(config)
text_formatter = TextFormatter()
summarizer = Summarizer(config)

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Recognition and Summarization</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .upload-form {
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
        }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background-color: #007bff;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        button:hover {
            background-color: #0056b3;
        }
        button:disabled {
            background-color: #6c757d;
            cursor: not-allowed;
        }
        .progress-container {
            margin-top: 20px;
            display: none;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
        }
        .progress-fill {
            height: 100%;
            background-color: #28a745;
            width: 0%;
            transition: width 0.3s;
        }
        .result-container {
            margin-top: 30px;
            display: none;
        }
        .summary-box {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 4px;
            border-left: 4px solid #007bff;
        }
        .error {
            color: #dc3545;
            background-color: #f8d7da;
            padding: 10px;
            border-radius: 4px;
            margin-top: 10px;
            display: none;
        }
        .loading {
            display: none;
            text-align: center;
            margin-top: 20px;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Audio Recognition and Summarization</h1>
        
        <form class="upload-form" id="uploadForm" enctype="multipart/form-data">
            <div class="form-group">
                <label for="audioFile">Upload Audio File:</label>
                <input type="file" id="audioFile" name="audio" accept="audio/*" required>
            </div>
            
            <button type="submit" id="submitBtn">Process Audio</button>
        </form>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Processing audio... This may take several minutes depending on file size.</p>
        </div>
        
        <div class="progress-container" id="progressContainer">
            <p>Progress:</p>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <p id="progressText">Initializing...</p>
        </div>
        
        <div class="error" id="errorContainer"></div>
        
        <div class="result-container" id="resultContainer">
            <h2>Summary</h2>
            <div class="summary-box" id="summaryResult"></div>
            
            <h2>Transcript</h2>
            <textarea id="transcriptResult" rows="15" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px;" readonly></textarea>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData();
            const audioFile = document.getElementById('audioFile').files[0];
            
            if (!audioFile) {
                showError('Please select an audio file');
                return;
            }
            
            formData.append('audio', audioFile);
            
            // Show loading indicators
            document.getElementById('loading').style.display = 'block';
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('errorContainer').style.display = 'none';
            document.getElementById('resultContainer').style.display = 'none';
            
            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Update progress to 100%
                    document.getElementById('progressFill').style.width = '100%';
                    document.getElementById('progressText').textContent = 'Completed!';
                    
                    // Display results
                    document.getElementById('summaryResult').textContent = data.summary;
                    document.getElementById('transcriptResult').textContent = data.transcript;
                    document.getElementById('resultContainer').style.display = 'block';
                } else {
                    showError(data.error || 'An error occurred during processing');
                }
            } catch (error) {
                showError('Network error: ' + error.message);
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('submitBtn').disabled = false;
            }
        });
        
        function showError(message) {
            const errorContainer = document.getElementById('errorContainer');
            errorContainer.textContent = message;
            errorContainer.style.display = 'block';
        }
        
        // Simulate progress updates (in a real app, this would come from the backend)
        function simulateProgress() {
            let progress = 0;
            const interval = setInterval(() => {
                if (progress >= 95) {
                    clearInterval(interval);
                    return;
                }
                progress += Math.random() * 15;
                if (progress > 95) progress = 95;
                
                document.getElementById('progressFill').style.width = progress + '%';
                document.getElementById('progressText').textContent = `Processing: ${Math.round(progress)}%`;
            }, 500);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Render the main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/process', methods=['POST'])
def process_audio():
    """Process uploaded audio file and return summary and transcript."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Create a temporary file to store the uploaded audio
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        file.save(temp_file.name)
        temp_filename = temp_file.name
    
    try:
        # Process audio and get transcript with speakers
        transcript_data = audio_processor.process_audio(temp_filename)
        
        # Format the transcript
        formatted_transcript = text_formatter.format_transcript(transcript_data)
        
        # Generate summary
        summary = summarizer.generate_summary(formatted_transcript)
        
        # Return the results
        return jsonify({
            'summary': summary,
            'transcript': formatted_transcript
        })
    
    except Exception as e:
        return jsonify({'error': f'Error processing audio: {str(e)}'}), 500
    
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)


@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)