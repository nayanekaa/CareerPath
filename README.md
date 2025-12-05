
**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`

### Backend (optional, recommended for RAG and verification)

This repo now includes a lightweight Python backend under `backend/` that implements RAG, agents, verification, and demo mode.

1. Start the backend (see `backend/README.md`).
2. Set `VITE_BACKEND_URL` in your `.env` or environment to `http://localhost:8000` so the frontend will call the local backend.

If no `OPENAI_API_KEY` is present the backend runs in demo mode using precomputed canned outputs.
