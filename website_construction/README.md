# Website Construction

A small starter web application for a personal portfolio site with a family tree module.

## Stack

- Backend: FastAPI + Python 3.11+
- Frontend: HTML, CSS, and browser-ready JavaScript with companion TypeScript source files under `frontend/assets`

## Canonical Family Data

- The canonical family dataset lives at `backend/data/family.json`.
- The backend reads that file on startup and all Family page edits now write back to that same file.
- Before each write, the backend creates `backend/data/family.json.bak`.
- Writes are atomic: the server writes `family.json.tmp` first and then replaces `family.json`.

## Family API

- `GET /api/family`
- `POST /api/family`
- `PUT /api/family/{id}`
- `GET /api/person/{id}`
- `GET /api/tree`

Delete is intentionally not exposed yet. The current schema stores spouse links by name rather than by id, so removing a person safely without risking ambiguous string references would need a broader schema cleanup first.

## Family Editing Auth

Family write endpoints are currently open for local development, so the Family page can save edits without a token.

## Run Locally

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open [http://127.0.0.1:8000/family](http://127.0.0.1:8000/family).

### Frontend Assets

The frontend is served by FastAPI from `frontend/`, so you do not need a separate frontend dev server for this project.

If you want to type-check the TypeScript source files:

```bash
cd frontend
npm install
npm run build
```

`npm run build` runs `tsc` as a type check against the `.ts` source files. The browser loads the committed `.js` files from `frontend/assets`.

## How Edit Persistence Works

1. The Family page fetches records from `GET /api/family`.
2. Clicking `Edit` opens an inline editor for that row, and `Add family member` opens a blank form.
3. Saving sends the updated record to `POST /api/family` or `PUT /api/family/{id}`.
4. The backend validates the payload, syncs reciprocal parent and child links, updates spouse name links when they point at an existing person record, writes a backup, and atomically replaces `backend/data/family.json`.
5. After a successful save, the page re-fetches family data and re-renders both the table and the tree without a full reload.

## Family Schema Assumptions

- `id` values are stable keys and use lowercase letters, numbers, and hyphens.
- Existing record ids are not editable through the update route because parent and child relationships are id-based.
- Names must stay unique across the dataset because spouse links currently resolve by name.
- `deathYear` may be empty or `null`.
- Some legacy ancestor ids in the current dataset do not yet have full person records. Those legacy unresolved references are preserved, but newly introduced invalid parent or child ids are rejected on writes.
- When linked parent ids are present, the backend refreshes the saved `father` and `mother` text fields from the ordered parent records.
