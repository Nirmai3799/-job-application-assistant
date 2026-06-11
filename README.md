# AI Job Application Assistant

Upload your resume and a job description — get tailored resume bullets and a cover letter in seconds.

## Problem it solves

Job hunting is repetitive and exhausting. For every application, you're expected to rewrite your resume bullets to match that company's language, write a fresh cover letter that doesn't sound generic, and figure out which of your skills to highlight — then do it all over again for the next one.

Most people either send the same generic application everywhere (low response rate) or spend 2+ hours tailoring each one (unsustainable at scale).

This tool automates the tailoring work. You bring your resume and the job — it handles the rewriting.

## What it does

1. **Reads your resume** — upload a PDF or paste the text
2. **Reads the job description** — paste a URL or the full text
3. **Generates a complete application package:**
   - Job analysis (required skills, culture, salary, remote policy)
   - 6 resume bullets rewritten to match the job's exact language and keywords
   - A personalized cover letter that doesn't sound like a template

## Tech stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [Anthropic Claude](https://anthropic.com) (`claude-sonnet-4-6`) | Job analysis + resume bullet generation |
| [OpenAI GPT](https://openai.com) (`gpt-4.1-mini`) | Cover letter generation |
| BeautifulSoup + requests | Scrapes job postings from URLs |
| PyPDF2 | Extracts text from uploaded PDF resumes |

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add your API keys**

Create a `.env` file in this folder:
```
ANTHROPIC_API_KEY=your-key-here
OPENAI_API_KEY=your-key-here
```

**3. Run**
```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

## Project structure

```
job-application-assistant/
├── app.py            # Main application
├── requirements.txt  # Dependencies
├── .env              # API keys (not committed)
└── README.md
```
