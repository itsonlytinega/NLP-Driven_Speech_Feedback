# NLP-Driven Speech Feedback System

A comprehensive Django-based web application that provides AI-powered speech analysis and personalized coaching. The system uses advanced NLP techniques (Whisper, BERT, spaCy) to analyze speech patterns, detect filler words, assess pronunciation, and recommend targeted practice drills.

## Project Overview

This system provides real-time speech analysis and feedback across multiple dimensions:

**Analysis Categories:**
- **Filler Word Detection** - Identifies and tags filler words (um, uh, like, you know, etc.) using spaCy NLP and regex patterns
- **Speech Quality Metrics** - Analyzes clarity, pacing, and fluency using BERT-based models
- **Pronunciation Analysis** - Enhanced pronunciation scoring with accent detection
- **Pacing Analysis** - Calculates words per minute (WPM) and identifies pacing issues
- **Root Cause Classification** - ML-based classification of speech issues (anxiety, stress, lack of confidence, poor skills)

**Models:**
- **Whisper** (OpenAI) - Speech-to-text transcription with word-level timestamps
- **BERT Fine-tuned** - Filler detection, clarity scoring, and pacing analysis
- **spaCy NLP** - Context-aware filler word detection and tagging
- **Hybrid BERT-Wav2Vec** - Advanced speech quality assessment

**Performance:**
- Real-time transcription with <2s latency
- Filler word detection accuracy: 85%+ with spaCy
- Multi-label classification for speech quality metrics
- Personalized drill recommendations based on analysis

## Features

### Core Functionality
-  **Speech Session Recording** - Upload or record audio directly in browser
-  **Comprehensive Analysis** - Filler words, pacing, clarity, and pronunciation scoring
-  **Personalized Drill Recommendations** - AI-powered drill suggestions based on analysis
-  **Progress Tracking** - Historical analysis and performance trends
-  **Secure Authentication** - Email verification, 2FA (TOTP), and WebAuthn passkeys
-  **REST API** - Full API access for speech sessions and analytics

### Drill Types
- **Filler Words** - Filler Zap Game, Filler Hunt Mirror, Silent Switcheroo
- **Pacing** - Metronome Rhythm, Timer Tag Team, Pause Powerup, Pause Pyramid
- **Pronunciation** - Vowel Vortex, Phonetic Puzzle, Pencil Precision, Echo Elimination
- **Fluency** - Shadow Superhero, Echo Chamber, Poem Echo, Slow Motion Story
- **Advanced** - Beat Drop Dialogue, Mirror Mimic, Word Swap Whirlwind

### Security Features
- Email-based authentication with verification
- Two-Factor Authentication (2FA) via TOTP
- WebAuthn passkey support
- Backup codes for account recovery
- Session-based and token-based API authentication

## Requirements

- **Python 3.12+** (required)
- **PostgreSQL** 12+ (database)
- **Redis** (optional, for caching and Celery)
- **FFmpeg** (for audio processing)
- **8GB+ RAM** recommended for ML models
- **GPU** (optional, for faster model inference)

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/itsonlytinega/NLP-Driven_Speech_Feedback.git
cd NLP-Driven_Speech_Feedback
```

### 2. Create and activate virtual environment

**Create virtual environment:**
```bash
python3.12 -m venv .venv
```

**Note:** Ensure you have Python 3.12 or later installed. Check your version:
```bash
python3 --version
```

**Activate virtual environment:**

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

**Note:** You should see `(.venv)` in your terminal prompt when activated.

### 3. Install dependencies

**Install base requirements:**
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**Install NLP-specific dependencies:**
```bash
pip install -r requirements_nlp.txt
```

**Install spaCy English model:**
```bash
python -m spacy download en_core_web_sm
```

### 4. Set up PostgreSQL database

Create a PostgreSQL database:
```sql
CREATE DATABASE verbalcoach_db;
CREATE USER verbalcoach_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE verbalcoach_db TO verbalcoach_user;
```

### 5. Configure environment variables

Create a `.env` file in the project root:
```env
# Database
DB_NAME=verbalcoach_db
DB_USER=verbalcoach_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Django
SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1

# Email (for verification and password reset)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USE_TLS=True
EMAIL_HOST_USER=your-email@gmail.com
EMAIL_HOST_PASSWORD=your-app-password

# Redis (optional, for caching)
REDIS_URL=redis://localhost:6379/0
```

### 6. Run database migrations

```bash
python manage.py migrate
```

### 7. Create superuser

```bash
python manage.py createsuperuser
```

### 8. Populate drills (optional)

```bash
python manage.py populate_drills
```

### 9. Collect static files

```bash
python manage.py collectstatic --noinput
```

### 10. Run the development server

```bash
python manage.py runserver
```

The application will be available at `http://127.0.0.1:8000`

## API Endpoints

### Authentication
- `POST /api/auth/register/` - User registration
- `POST /api/auth/login/` - User login (returns token)
- `POST /api/auth/logout/` - User logout

### Speech Sessions
- `GET /api/sessions/` - List user's speech sessions (with filtering and search)
- `POST /api/sessions/` - Create new speech session
- `GET /api/sessions/{id}/` - Retrieve specific session details
- `PUT /api/sessions/{id}/` - Update session
- `PATCH /api/sessions/{id}/` - Partial update
- `DELETE /api/sessions/{id}/` - Delete session
- `GET /api/sessions/analytics/` - Get user analytics and statistics

### Filtering & Search
- `?status=analyzed` - Filter by status (pending, analyzed, archived)
- `?date__gte=2024-01-01` - Filter by date range
- `?search=keyword` - Search in transcription and analysis
- `?ordering=-date` - Order by date, duration, filler_count, etc.

### Example API Usage

**Create a speech session:**
```bash
curl -X POST http://127.0.0.1:8000/api/sessions/ \
  -H "Authorization: Token your-token-here" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@recording.mp3" \
  -F "duration=120"
```

**Get session analytics:**
```bash
curl -X GET http://127.0.0.1:8000/api/sessions/analytics/ \
  -H "Authorization: Token your-token-here"
```

## Project Structure

```
NLP-Driven_Speech_Feedback/
├── coach/                          # Main application
│   ├── models.py                   # User, Feedback, Drill models
│   ├── views.py                    # View functions and class-based views
│   ├── forms.py                    # Django forms
│   ├── urls.py                     # URL routing
│   ├── utils.py                    # Speech analysis utilities
│   ├── admin.py                    # Django admin configuration
│   ├── templates/coach/            # HTML templates
│   │   ├── dashboard.html          # User dashboard
│   │   ├── drill_*.html            # Drill templates
│   │   └── ...
│   ├── migrations/                 # Database migrations
│   └── models/                     # ML model files (git-ignored)
│       ├── bert_speech/           # BERT model configs
│       └── hybrid_bert_wav2vec/    # Hybrid model configs
├── speech_sessions/                # Speech session app
│   ├── models.py                   # SpeechSession, DetectedFiller models
│   ├── views.py                    # Session views
│   ├── api.py                      # REST API ViewSets
│   ├── nlp_pipeline.py             # NLP analysis pipeline
│   ├── pronunciation_analyzer.py   # Pronunciation analysis
│   ├── serializers.py              # DRF serializers
│   ├── forms.py                    # Session forms
│   ├── admin.py                    # Admin interface
│   └── templates/                  # Session templates
├── verbalcoach/                    # Django project settings
│   ├── settings.py                 # Django configuration
│   ├── urls.py                     # Root URL configuration
│   ├── wsgi.py                     # WSGI application
│   └── asgi.py                     # ASGI application
├── static/                         # Static files (CSS, JS)
├── templates/                      # Base templates
├── scripts/                        # Utility scripts
│   ├── install_pronunciation_deps.py
│   ├── run_full_pipeline_v2.py
│   └── test_nlp_pipeline.py
├── data/                           # Training data (git-ignored)
├── media/                          # User-uploaded files (git-ignored)
├── manage.py                       # Django management script
├── requirements.txt                # Base dependencies
├── requirements_nlp.txt            # NLP/ML dependencies
└── README.md                       # This file
```

## Model Artifacts

Trained model files are **not stored in Git** due to their size (100MB+). The system includes:

- **BERT tokenizer configs** - Pre-configured tokenizers for speech analysis
- **Model architecture files** - JSON configs for model loading

**To use the full system:**
1. Models will be automatically downloaded/loaded when first used
2. Fine-tuned models can be placed in `coach/models/` directory
3. The system falls back to rule-based detection if models are unavailable

## Key Features Explained

### Filler Word Detection Module
- **spaCy NLP Detection** - Context-aware detection using part-of-speech tagging
- **Regex Fallback** - Pattern matching for common fillers
- **Database Storage** - Each detected filler is stored with position, timing, and context
- **Tagging System** - Filler words are tagged in transcripts with metadata

### Speech Quality Analysis
- **BERT-based Scoring** - Multi-dimensional quality assessment
- **Ensemble Methods** - Combines multiple models for robust predictions
- **Real-time Analysis** - Fast inference with caching support

### Drill Recommendation System
- **ML-based Recommendations** - Uses analysis scores to suggest drills
- **Personalized** - Considers user history and performance patterns
- **Multi-category** - Supports filler words, pacing, pronunciation, and fluency drills

## Hardware Requirements

- **Minimum:** 8GB RAM, CPU-only (slower inference)
- **Recommended:** 16GB+ RAM, GPU for faster model inference
- **Storage:** 5GB+ for models and dependencies
- **Network:** Internet connection for initial model downloads

## Troubleshooting

**Python version error?**
This project requires Python 3.12 or later:
```bash
python3 --version  # Should show 3.12.x or higher
```

**Database connection error?**
- Ensure PostgreSQL is running
- Check `.env` file has correct database credentials
- Verify database exists: `psql -l | grep verbalcoach_db`

**spaCy model not found?**
```bash
python -m spacy download en_core_web_sm
```

**FFmpeg not found?**
Install FFmpeg:
- **Windows:** `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
- **macOS:** `brew install ffmpeg`
- **Linux:** `sudo apt-get install ffmpeg`

**Audio transcription fails?**
- Check audio file format (supports MP3, WAV, M4A, WebM)
- Ensure file is not corrupted
- Verify Whisper model is loaded (check logs)

**Out of memory during analysis?**
- Reduce batch size in `coach/utils.py`
- Use smaller Whisper model (base instead of large)
- Enable model caching in settings

**Static files not loading?**
```bash
python manage.py collectstatic --noinput
```

**Migration errors?**
```bash
python manage.py makemigrations
python manage.py migrate
```

## Development

### Running Tests
```bash
python manage.py test
```

### Creating Migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

### Accessing Django Admin
1. Create superuser: `python manage.py createsuperuser`
2. Navigate to: `http://127.0.0.1:8000/admin/`
3. Login with superuser credentials

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Code Style** - Follow PEP 8 Python style guide
2. **Comments** - Explain **why**, not **what** (meaningful comments)
3. **Testing** - Add tests for new features
4. **Documentation** - Update README and docstrings for changes
5. **Commits** - Use conventional commit messages: `feat(scope): description`

## License

This project is part of an academic assignment. See repository for license details.

## Acknowledgments

- **OpenAI Whisper** - Speech transcription
- **Hugging Face Transformers** - BERT models and tokenizers
- **spaCy** - NLP processing
- **Django** - Web framework
- **Django REST Framework** - API framework

---

## Quick Start Summary

```bash
# 1. Clone and setup
git clone https://github.com/itsonlytinega/NLP-Driven_Speech_Feedback.git
cd NLP-Driven_Speech_Feedback
python3.12 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt
pip install -r requirements_nlp.txt
python -m spacy download en_core_web_sm

# 3. Setup database and environment
# Create PostgreSQL database and configure .env file

# 4. Run migrations
python manage.py migrate
python manage.py createsuperuser
python manage.py collectstatic --noinput

# 5. Start server
python manage.py runserver
```

Visit `http://127.0.0.1:8000` to access the application!
