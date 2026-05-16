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

app = Flask(__name__)

def merge_captions(transcript_list, max_words=80):
    merged = []
    buffer, start_time = [], None
    for item in transcript_list:
        words = item['text'].split()
        if not buffer:
            start_time = item['start']
        buffer += words
        if len(buffer) >= max_words:
            merged.append({
                "text": " ".join(buffer),
                "start": start_time
            })
            buffer = []
    if buffer:
        merged.append({"text": " ".join(buffer), "start": start_time})
    return merged


# 🚀 SPEED BOOST: Load models once at startup
print("🧠 Loading AI models, please wait...")
summarizer = pipeline("summarization", model="t5-small", device=-1)
asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
print("✅ AI models loaded successfully!")


# ----------------------------
# 1️⃣ Extract YouTube video ID
# ----------------------------
def extract_youtube_id(url):
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None


# ----------------------------
# 2️⃣ Get Transcript (with Timestamps)
# ----------------------------
def get_transcript_with_timestamps(youtube_url):
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
        print(f"⚠ No captions for {video_id}. Will fallback to Whisper.")
        return None, "No captions found"
    except Exception as e:
        print(f"🚨 Error fetching captions: {e}")
        return None, str(e)


# ----------------------------
# 3️⃣ Stream & Transcribe with Whisper
# ----------------------------
def transcribe_with_whisper(youtube_url, progress_callback):
    progress_callback({"status": "Downloading audio with yt-dlp...", "progress": 20})

    try:
        # Download audio first (faster + more stable)
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': 'temp_audio.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            audio_path = ydl.prepare_filename(info).rsplit('.', 1)[0] + '.wav'

        progress_callback({"status": "Transcribing audio with Whisper...", "progress": 60})
        print(f"🎧 Transcribing {audio_path}...")

        # --- Run Whisper ---
        result = asr(audio_path, return_timestamps=True, generate_kwargs={"language": "en"})

        # Delete temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)
            print("🧹 Temporary audio file deleted.")

        print("✅ Whisper transcription completed successfully!")

        # Parse result into transcript list
        transcript_list = []
        for chunk in result.get('chunks', []):
            transcript_list.append({
                'text': chunk['text'].strip(),
                'start': chunk['timestamp'][0],
                'duration': chunk['timestamp'][1] - chunk['timestamp'][0]
            })

        full_text = " ".join([item['text'] for item in transcript_list])
        return transcript_list, full_text

    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed: {e}")


# ----------------------------
# 4️⃣ Summarize Text
# ----------------------------
def summarize_text(text, progress_callback):
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
    """
    Extracts key sentences (moments) from a transcript using LSA summarization,
    aligns them to timestamps, and ensures stable output even with merged captions.
    """
    if not transcript_list:
        return []

    # ✅ Merge captions into larger chunks for better context
    merged_captions = merge_captions(transcript_list, max_words=80)
    full_text = " ".join([item["text"] for item in merged_captions])

    # --- Use SUMY LSA summarizer to get key sentences ---
    parser = PlaintextParser.from_string(full_text, Tokenizer("english"))
    lsa_summarizer = LsaSummarizer()
    summary_sentences = list(lsa_summarizer(parser.document, num_moments))

    key_moments = []

    # --- Match summarized sentences back to timestamps ---
    for sentence in summary_sentences:
        sentence_str = str(sentence).strip()
        best_match = None
        best_overlap = 0

        for item in merged_captions:
            # Simple overlap metric (how many words overlap)
            overlap = len(set(sentence_str.split()) & set(item['text'].split()))
            if overlap > best_overlap:
                best_overlap = overlap
                best_match = item

        if best_match:
            key_moments.append({
                "timestamp": int(best_match['start']),
                "text": sentence_str
            })

    # --- Sort by timestamp ---
    key_moments.sort(key=lambda x: x['timestamp'])
    return key_moments

# ----------------------------
# 6️⃣ Convert Summary to Notes
# ----------------------------
def generate_notes_from_summary(summary_text):
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

            # Step 1: Fetch transcript
            yield f"data: {json.dumps({'status': 'Fetching transcript...', 'progress': 5})}\n\n"
            transcript_list, error = get_transcript_with_timestamps(youtube_url)

            if not transcript_list:
                # Step 2: Transcribe audio with Whisper (with terminal progress logs)
                def terminal_progress(update):
                    print(f"🌀 {update['status']} ({update['progress']}%)")

                transcript_list, full_transcript_text = transcribe_with_whisper(
                    youtube_url, terminal_progress
                )
            else:
                full_transcript_text = " ".join([item['text'] for item in transcript_list])

            if not full_transcript_text:
                yield f"data: {json.dumps({'error': 'Could not retrieve transcript.'})}\n\n"
                return

            # Step 3: Summarize
            yield f"data: {json.dumps({'status': 'Generating summary...', 'progress': 70})}\n\n"
            summary_text = summarize_text(full_transcript_text, lambda update: None)

            # Step 4: Extract key moments
            yield f"data: {json.dumps({'status': 'Extracting key moments...', 'progress': 95})}\n\n"
            key_moments = extract_key_moments(transcript_list)

            # Step 5: Generate notes
            bullet_notes = generate_notes_from_summary(summary_text)

            result = {
                "summary": summary_text,
                "notes": bullet_notes,
                "transcript": full_transcript_text,
                "moments": key_moments
            }

            yield f"data: {json.dumps({'result': result, 'progress': 100})}\n\n"

        except Exception as e:
            print(f"🚨 CRITICAL ERROR: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


if __name__ == "__main__":
    app.run(debug=True)
