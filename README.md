---
title: AI Job Application Assistant
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# AI Job Application Assistant — ATS Optimizer

Upload your resume and a job description and get an ATS-optimized resume, skill gap plan, and personalized cover letter in seconds.

## Problem it solves

Recruiters at most companies use ATS (Applicant Tracking System) software to automatically filter resumes before a human ever reads them. A resume that misses key phrases from the job description gets rejected automatically, even if the candidate is a strong fit.

Most people either send the same generic resume everywhere (low pass rate) or spend 2+ hours manually tailoring each one (unsustainable when applying to 20+ jobs).

This tool automates the tailoring work. You bring your resume and the job — it handles the ATS optimization.

## What it does

1. **ATS Score** — analyzes how well your resume matches the job, gives a 0–100 score, and shows exactly which keywords you're matched and missing
2. **Optimized Resume** — rewrites your full resume to incorporate missing keywords naturally, following ATS-friendly formatting rules; downloadable as PDF
3. **Before vs After** — re-runs the ATS check on the optimized resume so you can see the score improvement and a pass/fail verdict
4. **AI Chat Console** — refine the optimized resume through a chat interface before downloading; ask for specific changes or revert to your original style
5. **Skill Gap Plan** — for skills still missing after optimization, shows what to build, estimated score points each skill adds, and a projected ATS score if you complete them
6. **Cover Letter** — generates a personalized cover letter matched to the specific company and role

## Tech stack

| Tool | Purpose |
|---|---|
| [Streamlit](https://streamlit.io) | Web UI |
| [Anthropic Claude](https://anthropic.com) (`claude-sonnet-4-6`) | ATS analysis (structured outputs via Tool Use), resume rewrite + refinement (streaming), skill gap analysis |
| [OpenAI GPT](https://openai.com) (`gpt-4.1-mini`) | Cover letter generation (streaming) |
| BeautifulSoup + requests | Scrapes job postings from URLs |
| PyPDF2 | Extracts text from uploaded PDF resumes |
| fpdf2 | Generates downloadable PDF of the optimized resume |
| Pushover | Push notification to phone on every usage |

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
PUSHOVER_USER=your-pushover-user-key
PUSHOVER_TOKEN=your-pushover-app-token
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
├── requirements.txt  # Python dependencies
├── .env              # API keys (not committed to git)
├── .gitignore
└── README.md
```

## How the ATS analysis works

Claude uses **Tool Use** (structured outputs) to analyze the resume and job description. Rather than returning free-form text that we'd have to parse, it is forced to call a schema-defined tool — meaning the response is always a valid, typed Python dict with fields like `ats_score`, `matched_keywords`, `missing_keywords`, etc. No JSON parsing, no regex, no failures.

The same structured output approach is used for the skill gap analysis — each gap comes back with a skill name, suggested project, and estimated score points gained, guaranteed by schema.
