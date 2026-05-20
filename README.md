# 🎥 AI Powered Video Summarizer

## 📌 Project Overview

AI Video Summarizer is a third-year Machine Learning mini project that automatically generates summaries from YouTube videos using Artificial Intelligence and Natural Language Processing (NLP).

The system extracts transcripts from YouTube videos, processes the text using transformer-based AI models, and generates a concise summary along with important key moments.

If captions are unavailable, the system uses Whisper speech recognition to transcribe audio automatically.

---

# 🚀 Features

* 🔗 Accepts YouTube video links
* 📝 Automatically extracts video transcript
* 🤖 AI-powered text summarization using Hugging Face Transformers
* 🎤 Whisper-based speech-to-text fallback
* ⏱ Key moment extraction from video content
* ⚡ Fast processing with cached transcripts
* 🌐 Simple and user-friendly web interface

---

# 🛠 Technologies Used

## Frontend

* HTML
* CSS
* JavaScript

## Backend

* Python
* Flask

## Machine Learning / AI

* Hugging Face Transformers
* T5-small Summarization Model
* OpenAI Whisper Tiny Model
* NLP (Natural Language Processing)

## Other Libraries

* yt-dlp
* youtube-transcript-api
* pydub
* nltk
* sumy
* ffmpeg

---

# 📂 Project Structure

```bash
video_summarizer/
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── summarizer.py
│
└── README.md
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone <repository-link>
cd video_summarizer
```

## 2️⃣ Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
```

### Activate Virtual Environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / Mac

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Required Dependencies

```bash
pip install flask transformers youtube-transcript-api yt-dlp pydub nltk sumy torch
```

---

## 4️⃣ Install FFmpeg

FFmpeg is required for audio processing.

### Windows

1. Download FFmpeg
2. Add FFmpeg to system PATH

### Linux

```bash
sudo apt install ffmpeg
```

### Mac

```bash
brew install ffmpeg
```

---

# ▶️ Running the Project

Run the Flask application:

```bash
python summarizer.py
```

Open browser and visit:

```bash
http://127.0.0.1:5000
```

---

# 🧠 How the System Works

1. User enters YouTube video URL
2. System extracts video ID
3. Transcript is fetched using YouTube Transcript API
4. If transcript is unavailable:

   * Audio is streamed
   * Whisper model converts speech to text
5. AI summarization model generates concise summary
6. Key moments are extracted using NLP
7. Results are displayed on web page

---

# 📸 Future Enhancements

* Multi-language support
* PDF summary export
* Video keyword extraction
* Real-time live video summarization
* Mobile application support
* User login and history tracking

---

# 🎯 Learning Outcomes

This project helps students understand:

* Machine Learning integration in web applications
* NLP and text summarization
* Speech recognition using Whisper
* Flask backend development
* API integration
* YouTube data processing

---

# 👨‍💻 Project Type

* Third Year Engineering Project
* Domain: Machine Learning & NLP
* Category: AI-Based Video Processing System

---

# 📜 Conclusion

AI Video Summarizer reduces the time required to understand long videos by generating quick summaries automatically using Machine Learning and NLP techniques. The project demonstrates practical implementation of AI models in real-world applications.

---

# 📄 License

This project is developed for educational purposes.
