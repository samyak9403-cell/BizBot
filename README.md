# BizBot

## Backend (Mistral)

### Setup
1. Create and activate a virtual environment.
2. Install dependencies:
	`pip install mistralai python-dotenv`
3. Create a `.env` file from the example:
	- Windows PowerShell: `Copy-Item .env.example .env`
	- macOS/Linux: `cp .env.example .env`
4. Add your key to `.env`:
	`MISTRAL_API_KEY=your_key_here`

### Run
`python backend/index.py`