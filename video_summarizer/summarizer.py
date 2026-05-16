from flask import Flask, render_template, request, jsonify, Response
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pydub import AudioSegment
import yt_dlp
import os
import re
import math
import json
import nltk
import io
import subprocess
import tempfile

# --- SUMY for Key Moment Extraction ---
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# --- NLTK Setup (for sumy) ---
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 NLTK 'punkt' package not found. Downloading...")
    nltk.download('punkt')
    print("✅ Download complete.")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("📥 NLTK 'punkt_tab' package not found. Downloading...")
    nltk.download('punkt_tab')
    print("✅ Download complete.")


app = Flask(__name__)

# 🚀 SPEED BOOST: Load models once at startup for huge performance gain
print("🧠 Loading AI models, please wait...")
summarizer = pipeline("summarization", model="t5-small", device=-1)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
print("✅ AI models loaded successfully!")


# ----------------------------
# 1️⃣ Extract YouTube video ID
# ----------------------------
def extract_youtube_id(url):
    """Extracts the 11-character YouTube video ID from a URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

# ----------------------------
# 2️⃣ Get Transcript (with Timestamps)
# ----------------------------
def get_transcript_with_timestamps(youtube_url):
    """Fetches transcript from YouTube captions, including timestamps."""
    video_id = extract_youtube_id(youtube_url)
    if not video_id:
        return None, "Invalid YouTube URL"
    
    transcript_cache_path = f"{video_id}_transcript.json"
    if os.path.exists(transcript_cache_path):
        print(f"📄 Using cached transcript for {video_id}")
        with open(transcript_cache_path, 'r') as f:
            return json.load(f), None

    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        print(f"✅ Captions found for {video_id}")
        with open(transcript_cache_path, 'w') as f:
            json.dump(transcript_list, f)
        return transcript_list, None
    except (TranscriptsDisabled, NoTranscriptFound):
        print(f"⚠️ No captions for {video_id}. Will fallback to Whisper.")
        return None, "No captions found"
    except Exception as e:
        print(f"🚨 Error fetching captions: {e}")
        return None, str(e)

# ----------------------------
# 3️⃣ (IMPROVED) Stream & Transcribe with Whisper
# ----------------------------
def transcribe_with_whisper(youtube_url, progress_callback):
    """
    Streams audio directly from YouTube and transcribes it in chunks for maximum speed.
    Includes robust error handling for the streaming process.
    """
    progress_callback({"status": "Initializing audio stream...", "progress": 15})
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        audio_url = info['url']
        duration = info.get('duration', 0)

    # Use ffmpeg to stream the audio and process it in chunks
    ffmpeg_cmd = [
        'ffmpeg',
        # FIX 1: Add a user-agent to mimic a browser and prevent getting blocked
        '-user_agent', 'Mozilla/5.0', 
        '-i', audio_url,
        '-f', 's16le',
        '-ac', '1',
        '-ar', '16000',
        '-'
    ]
    
    process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    transcript_list = []
    total_processed_seconds = 0
    chunk_duration_s = 30
    chunk_size = 16000 * 2 * chunk_duration_s

    progress_callback({"status": "Transcribing in real-time...", "progress": 25})

    while True:
        in_bytes = process.stdout.read(chunk_size)
        if not in_bytes:
            break
        
        audio_chunk = AudioSegment(data=in_bytes, sample_width=2, frame_rate=16000, channels=1)
        
        with tempfile.NamedTemporaryFile(dir='.', suffix=".mp3", delete=True) as temp_file:
            audio_chunk.export(temp_file.name, format="mp3")
            result = asr(temp_file.name, return_timestamps=True)
            
            for chunk_data in result['chunks']:
                start_time = total_processed_seconds + chunk_data['timestamp'][0]
                end_time = total_processed_seconds + chunk_data['timestamp'][1]
                transcript_list.append({
                    'text': chunk_data['text'].strip(),
                    'start': start_time,
                    'duration': end_time - start_time
                })
        
        total_processed_seconds += chunk_duration_s
        
        if duration > 0:
            progress = 25 + int(45 * (total_processed_seconds / duration))
            progress_callback({"status": "Transcribing in real-time...", "progress": min(progress, 69)})

    # FIX 2: Capture and check for errors from ffmpeg
    stderr_output = process.stderr.read().decode('utf-8')
    process.wait()

    if process.returncode != 0:
        print(f"🚨 FFmpeg Error:\n{stderr_output}")
        raise RuntimeError("FFmpeg failed to process the audio stream.")
        
    return transcript_list, " ".join([item['text'] for item in transcript_list])


# ----------------------------
# 4️⃣ Summarize Text
# ----------------------------
def summarize_text(text, progress_callback):
    """Summarizes text using the pre-loaded model."""
    text_to_summarize = "summarize: " + text
    
    words = text_to_summarize.split()
    max_chunk_size = 400 
    chunks = [" ".join(words[i:i+max_chunk_size]) for i in range(0, len(words), max_chunk_size - 50)]
    
    summaries = []
    total_chunks = len(chunks)
    
    for i, chunk in enumerate(chunks):
        progress = 70 + int(20 * (i + 1) / total_chunks)
        progress_callback({"status": f"Summarizing chunk {i+1}/{total_chunks}...", "progress": progress})
        
        input_len = len(chunk.split())
        summary = summarizer(chunk, max_length=max(40, input_len // 4), min_length=20, do_sample=False)
        summaries.append(summary[0]['summary_text'])

    return " ".join(summaries)

# ----------------------------
# 5️⃣ Extract Key Moments
# ----------------------------
def extract_key_moments(transcript_list, num_moments=5):
    """Uses Sumy LSA to find the most important sentences/moments."""
    full_text = " ".join([item['text'] for item in transcript_list])
    if not full_text: return [] # Handle empty transcript
    
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    lsa_summarizer = LsaSummarizer()
    
    summary_sentences = lsa_summarizer(parser.document, num_moments)
    
    key_moments = []
    for sentence in summary_sentences:
        sentence_str = str(sentence)
        for item in transcript_list:
            if sentence_str in item['text']:
                key_moments.append({
                    "timestamp": int(item['start']),
                    "text": sentence_str
                })
                break
    
    key_moments.sort(key=lambda x: x['timestamp'])
    return key_moments

# ----------------------------
# 6️⃣ Convert Summary to Notes
# ----------------------------
def generate_notes_from_summary(summary_text):
    """Converts a paragraph summary into bullet points."""
    sentences = re.split(r'(?<=[.?!])\s+', summary_text)
    bullets = [f"• {s.strip()}" for s in sentences if s.strip()]
    return "\n".join(bullets)

# ----------------------------
# Flask Routes
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process_video():
    data = request.get_json()
    youtube_url = data.get("url")

    def generate():
        try:
            if not youtube_url or not extract_youtube_id(youtube_url):
                yield f"data: {json.dumps({'error': 'Please provide a valid YouTube URL.'})}\n\n"
                return

            def progress_callback(update):
                yield f"data: {json.dumps(update)}\n\n"
            
            yield f"data: {json.dumps({'status': 'Fetching transcript...', 'progress': 5})}\n\n"
            transcript_list, error = get_transcript_with_timestamps(youtube_url)
            
            if not transcript_list:
                transcript_list, full_transcript_text = transcribe_with_whisper(youtube_url, progress_callback)
            else:
                full_transcript_text = " ".join([item['text'] for item in transcript_list])
            
            if not full_transcript_text:
                 yield f"data: {json.dumps({'error': 'Could not retrieve transcript.'})}\n\n"
                 return

            yield f"data: {json.dumps({'status': 'Generating summary...', 'progress': 70})}\n\n"
            summary_text = summarize_text(full_transcript_text, progress_callback)
            
            yield f"data: {json.dumps({'status': 'Extracting key moments...', 'progress': 95})}\n\n"
            key_moments = extract_key_moments(transcript_list)

            bullet_notes = generate_notes_from_summary(summary_text)

            result = {
                "summary": summary_text,
                "notes": bullet_notes,
                "transcript": full_transcript_text,
                "moments": key_moments
            }
            yield f"data: {json.dumps({'result': result, 'progress': 100})}\n\n"

        except Exception as e:
            print(f"🚨🚨 CRITICAL ERROR: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)