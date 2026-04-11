# Website Construction

A small starter web application for a personal portfolio site with a family tree module.

## Stack

- Backend: FastAPI + Python 3.11+
- Frontend: HTML, CSS, TypeScript-style modular scripts with browser-ready JavaScript output

## Run the app

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000).

## Notes

- Family data lives in `backend/data/family.json`
- The API is fully data-driven
- Frontend pages are served by FastAPI
