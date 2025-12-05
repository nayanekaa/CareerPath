<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1eeqT8tlLq1Ih_JR699H1eZ-pSMRYc57i

## Run Locally

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
