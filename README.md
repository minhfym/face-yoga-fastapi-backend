# Face Yoga Analysis API

Advanced facial analysis backend using FastAPI, MediaPipe, and OpenCV.

## Features

- 🔍 **468-point facial landmark detection** with MediaPipe
- 📊 **Advanced wrinkle analysis** using OpenCV
- ⚖️ **Facial symmetry assessment**
- 🎯 **Contour analysis** for facial definition
- 🔐 **JWT authentication** with secure password hashing
- 🗄️ **PostgreSQL database** with SQLAlchemy ORM

## Tech Stack

### Backend
- FastAPI - Modern web framework
- SQLAlchemy - Database ORM
- Python-Jose - JWT authentication
- Passlib - Password hashing
- Python-multipart - File uploads
- Uvicorn - ASGI server

### ML/Computer Vision
- OpenCV - Image processing
- MediaPipe - Facial landmark detection
- NumPy - Numerical operations
- scikit-image - Advanced image processing

## Deployment

This API is configured for Railway deployment with automatic PostgreSQL database setup.

### Environment Variables

\`\`\`env
NEON_DATABASE_URL=postgresql://...  # Automatically provided by Railway
JWT_SECRET=your-secret-key
CORS_ORIGINS=https://your-frontend.com
\`\`\`

## API Endpoints

- `GET /` - API status
- `GET /health` - Health check
- `POST /api/auth/login` - User authentication
- `POST /api/auth/register` - User registration
- `POST /api/analysis/upload-and-analyze` - Image analysis
- `GET /api/clients/` - Client management

## Local Development

\`\`\`bash
pip install -r requirements.txt
uvicorn main:app --reload
\`\`\`

Visit `http://localhost:8000/docs` for interactive API documentation.
\`\`\`

Update the requirements.txt to be Railway-optimized:
