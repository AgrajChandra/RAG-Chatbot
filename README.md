# RAG Chatbot

A chatbot application that uses Retrieval-Augmented Generation (RAG) to provide accurate responses about EOXS products, daily updates, and employee information.

## Features

- Interactive chat interface
- Daily updates submission form
- Context-aware responses using RAG
- Real-time response streaming
- Beautiful and modern UI

## Prerequisites

- Node.js (v14 or higher)
- Python (3.8 or higher)
- Google API Key for Gemini

## Setup

### Backend Setup

1. Navigate to the Backend directory:
   ```bash
   cd Backend
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a .env file in the Backend directory:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. Start the backend server:
   ```bash
   python server.py
   ```

### Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000

## Usage

1. Open http://localhost:3000 in your browser
2. Use the chat interface to ask questions about:
   - Products and features
   - Daily team updates
   - Employee information
3. Use the Daily Update form to submit new team updates

## API Endpoints

- POST `/api/chat`: Send chat messages
  ```json
  {
    "message": "What are EOXS products?"
  }
  ```

- POST `/api/daily-update`: Submit daily updates
  ```json
  {
    "date": "2024-03-14",
    "team": "Engineering",
    "sub_team": "Frontend",
    "project": "RAG Chatbot",
    "present_members": "John, Jane",
    "summary": "Completed chat interface",
    "tasks_completed": ["UI implementation", "API integration"],
    "blockers": ["None"],
    "next_steps": "Start testing phase"
  }
  ``` 
