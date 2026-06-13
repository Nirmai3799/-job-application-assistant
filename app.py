"""
AI Job Application Assistant — ATS Optimizer
============================================
Upload your resume + a job description and get:
  1. ATS compatibility score with matched / missing keywords
  2. A fully rewritten ATS-optimized resume (PDF download)
  3. A personalized cover letter

Technical highlights:
  - Claude structured outputs via Tool Use (guaranteed schema, no JSON parsing)
  - Claude streaming for resume rewrite
  - OpenAI streaming for cover letter
  - Pushover push notification on every usage
  - PDF export via fpdf2 (Helvetica + unicodedata for safe encoding)

Run:  streamlit run app.py
"""

import io
import os
import re
import tempfile
import unicodedata

import anthropic
import PyPDF2
import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fpdf import FPDF
from openai import OpenAI


load_dotenv()

# Models
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4.1-mini"

# Pushover credentials (loaded from .env or hardcoded as fallback)
PUSHOVER_USER  = os.getenv("PUSHOVER_USER",  "utn1zi48yfimopfuvjp75d4nmo5qyp")
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN", "ai8hzryqf8dsry7bwq59j8ynfij8gr")


@st.cache_resource
def get_clients() -> tuple[anthropic.Anthropic, OpenAI]:
    """Initialize and cache API clients (called once per session)."""
    claude = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return claude, openai


claude_client, openai_client = get_clients()


def send_pushover(title: str, message: str) -> None:
    """
    Send a push notification via Pushover API.
    Silently ignores failures so the app never breaks due to notification issues.
    """
    try:
        requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token":   PUSHOVER_TOKEN,
                "user":    PUSHOVER_USER,
                "title":   title,
                "message": message,
            },
            timeout=5,
        )
    except Exception:
        pass  


def read_pdf(file_bytes: bytes) -> str:
    """Extract plain text from a PDF file's bytes."""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def scrape_job_url(url: str, max_chars: int = 8000) -> tuple[str | None, str | None]:
    """
    Scrape and clean job posting text from a URL.
    Returns (text, error_message). error_message is None on success.
    """
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            timeout=15,
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        lines = [
            line.strip()
            for line in soup.get_text("\n", strip=True).split("\n")
            if len(line.strip()) > 5
        ]
        return "\n".join(lines)[:max_chars], None
    except Exception as exc:
        return None, str(exc)


def to_ascii(text: str) -> str:
    """
    Normalize Unicode to closest ASCII equivalent.
    Example: é -> e, — -> -, smart quotes -> straight quotes.
    Required because fpdf2's built-in Helvetica only supports latin-1.
    """
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )


def strip_markdown(line: str) -> str:
    """Remove markdown formatting markers from a line, keeping visible text."""
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # [text](url) -> text
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"\*(.+?)\*",     r"\1", line)
    line = re.sub(r"_[^_\n]*_",     "",    line)
    return line.strip()


def build_pdf(markdown_text: str) -> bytes:
    """
    Render the optimized resume markdown to a clean PDF.

    Expected markdown format (what Claude outputs):
      # Full Name                → large centered name header
      plain contact line         → small gray contact info
      ## SECTION NAME            → bold blue section header with rule
      ### Company | Role | Dates → bold job/entry header
      - bullet text              → indented bullet with bullet character
      plain text                 → normal body paragraph

    Uses Helvetica (fpdf2 built-in) + to_ascii() for safe encoding.
    """
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()
    pdf.set_margins(18, 18, 18)
    W = pdf.w - pdf.l_margin - pdf.r_margin  # usable width

    def h1(text: str) -> None:
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(17, 24, 39)
        pdf.set_x(pdf.l_margin)
        pdf.cell(W, 10, to_ascii(text), ln=1, align="C")

    def contact(text: str) -> None:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(107, 114, 128)
        pdf.set_x(pdf.l_margin)
        pdf.cell(W, 5, to_ascii(text), ln=1, align="C")

    def h2(text: str) -> None:
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(37, 99, 235)
        pdf.set_x(pdf.l_margin)
        pdf.cell(W, 6, to_ascii(text).upper(), ln=1)
        pdf.set_draw_color(191, 219, 254)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
        pdf.ln(1)

    def h3(text: str) -> None:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 10)
        pdf.set_text_color(31, 41, 55)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 6, to_ascii(text))

    def bullet(text: str) -> None:
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(55, 65, 81)
        # Indent bullet by 4mm; remaining width shrinks accordingly
        pdf.set_x(pdf.l_margin + 4)
        pdf.multi_cell(W - 4, 5, to_ascii("- " + text))

    def body(text: str) -> None:
        pdf.set_font("Helvetica", "", 9.5)
        pdf.set_text_color(55, 65, 81)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(W, 5, to_ascii(text))

    # True only between the "# Name" line and the first "## Section" header.
    # Any plain-text line in that window is contact info; everything else is body.
    in_contact_zone = False

    for raw in markdown_text.split("\n"):
        line = raw.strip()

        if not line:
            pdf.ln(2)

        elif line.startswith("# "):
            h1(strip_markdown(line[2:]))
            in_contact_zone = True   # next plain-text lines are contact info

        elif line.startswith("## "):
            in_contact_zone = False  # first section header closes the contact zone
            h2(strip_markdown(line[3:]))

        elif line.startswith("### "):
            h3(strip_markdown(line[4:]))

        elif line.startswith(("- ", "* ")):
            in_contact_zone = False
            bullet(strip_markdown(line[2:]))

        elif line.startswith("---"):
            pdf.set_draw_color(229, 231, 235)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.l_margin + W, pdf.get_y())
            pdf.ln(2)

        else:
            if in_contact_zone:
                contact(strip_markdown(line))
            else:
                body(strip_markdown(line))

    # Write to a real temp file and read back — avoids all bytearray/str
    # conversion ambiguities across fpdf2 versions. File output path in fpdf2
    # is the most battle-tested route to valid PDF bytes.
    fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
    os.close(fd)
    try:
        pdf.output(tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if not data.startswith(b"%PDF"):
        raise ValueError(f"fpdf2 produced invalid output: {data[:30]!r}")
    return data


def score_color(score: int) -> str:
    """Return the CSS class name for the ATS score circle."""
    if score >= 75:
        return "score-green"
    if score >= 50:
        return "score-amber"
    return "score-red"


def score_verdict(score: int) -> str:
    """Return a human-readable verdict for an ATS score."""
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Partial match - needs work"
    return "Low match - likely rejected"


ATS_TOOL = {
    "name": "ats_report",
    "description": (
        "Report the ATS (Applicant Tracking System) compatibility analysis "
        "of a resume against a job description."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "ats_score": {
                "type": "integer",
                "description": (
                    "Overall ATS compatibility score from 0 to 100. "
                    "90-100: near-perfect, 75-89: good, 50-74: partial, 0-49: poor."
                ),
            },
            "matched_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Keywords present in both the resume and the job description.",
            },
            "missing_keywords": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Required keywords from the job that are completely absent in the resume.",
            },
            "nice_to_have_missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Preferred/bonus skills from the job not found in the resume.",
            },
            "issues": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Specific ATS red flags: missing sections, weak phrasing, formatting problems.",
            },
            "strengths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "What the resume already does well for this specific role.",
            },
        },
        "required": [
            "ats_score",
            "matched_keywords",
            "missing_keywords",
            "nice_to_have_missing",
            "issues",
            "strengths",
        ],
    },
}


SKILL_GAP_TOOL = {
    "name": "skill_gap_plan",
    "description": (
        "Produce a learning roadmap for skills the candidate is genuinely missing "
        "relative to the job requirements, based strictly on their existing resume."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "current_score": {
                "type": "integer",
                "description": "The candidate's current ATS score (0-100).",
            },
            "projected_score_if_all_gaps_filled": {
                "type": "integer",
                "description": (
                    "Estimated ATS score if the candidate completed all suggested "
                    "projects and added all missing skills to their resume."
                ),
            },
            "gaps": {
                "type": "array",
                "description": "One entry per genuinely missing required skill.",
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {
                            "type": "string",
                            "description": "The missing skill or technology.",
                        },
                        "why_it_matters": {
                            "type": "string",
                            "description": (
                                "In 1-2 sentences: why this specific job needs it "
                                "and how frequently it appears in the JD."
                            ),
                        },
                        "suggested_project": {
                            "type": "string",
                            "description": (
                                "A concrete, buildable project the candidate can do "
                                "to gain this skill — specific enough to start today. "
                                "Must relate to the candidate's existing background."
                            ),
                        },
                        "time_to_learn": {
                            "type": "string",
                            "description": "Realistic time estimate, e.g. '2-3 weekends' or '4 weeks'.",
                        },
                        "score_points_gained": {
                            "type": "integer",
                            "description": (
                                "Estimated ATS score increase from adding just this skill (1-20)."
                            ),
                        },
                    },
                    "required": [
                        "skill",
                        "why_it_matters",
                        "suggested_project",
                        "time_to_learn",
                        "score_points_gained",
                    ],
                },
            },
        },
        "required": ["current_score", "projected_score_if_all_gaps_filled", "gaps"],
    },
}


def analyze_ats(resume_text: str, job_text: str) -> dict:
    """
    Run ATS analysis using Claude with structured Tool Use output.

    Claude is forced to call the `ats_report` tool, which guarantees the
    response always matches our schema — no JSON parsing needed.

    Returns a dict with keys defined in ATS_TOOL['input_schema'].
    """
    prompt = (
        "You are an ATS (Applicant Tracking System) expert with deep knowledge "
        "of how enterprise recruiting software scores resumes.\n\n"
        "Analyze how well the resume below matches the job description. "
        "Be precise about which exact keywords are present or missing — "
        "ATS systems do literal string matching, not semantic matching.\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Resume:\n{resume_text}"
    )

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        tools=[ATS_TOOL],
        tool_choice={"type": "tool", "name": "ats_report"},  # force the tool call
        messages=[{"role": "user", "content": prompt}],
    )

    # With tool_choice forced, content[0] is always a ToolUseBlock
    # whose .input is already a validated Python dict — no parsing needed.
    # Claude occasionally returns integer fields as strings despite the schema;
    # coerce ats_score here so the UI never has to worry about it.
    result = response.content[0].input
    result["ats_score"] = int(result.get("ats_score") or 0)
    return result


def stream_optimized_resume(resume_text: str, job_text: str, ats: dict):
    """
    Rewrite the resume using Claude to maximize ATS score.

    Strict honesty rule: only rephrase/reorder what's already in the original.
    Never add new jobs, projects, skills, or certifications the candidate didn't have.
    Missing keywords that have no honest hook in the existing experience are left out —
    those go to the Skill Gap Plan tab instead.

    Yields text chunks for live streaming display.
    """
    matched      = ", ".join(ats.get("matched_keywords", []))
    issue_list   = "\n".join(f"  - {i}" for i in ats.get("issues", []))

    prompt = (
        "You are an expert resume writer and ATS optimization specialist.\n\n"
        "Rewrite the candidate's resume into a professional, recruiter-ready format "
        "while using ONLY the content from the original resume.\n\n"

        "━━━ MARKDOWN FORMAT (the PDF renderer requires this exactly) ━━━\n"
        "# Full Name\n"
        "email | phone | LinkedIn | location  (plain contact line, no heading)\n"
        "## PROFESSIONAL SUMMARY\n"
        "2-3 sentence summary paragraph.\n"
        "## WORK EXPERIENCE\n"
        "### Company Name | Job Title | Start Month Year – End Month Year\n"
        "- Achievement bullet\n"
        "## TECHNICAL SKILLS\n"
        "### Languages\n"
        "Python, Java, ...\n"
        "### Frameworks & Libraries\n"
        "React, PyTorch, ...\n"
        "### Tools & Platforms\n"
        "Docker, AWS, ...\n"
        "## EDUCATION\n"
        "### University Name | Degree | Graduation Year\n"
        "- Relevant coursework or honours if present\n"
        "## PROJECTS  (only if the original has projects)\n"
        "### Project Name | Tech Stack Used\n"
        "- What it does and what you built\n"
        "## CERTIFICATIONS  (only if the original has certifications)\n"
        "- Certification name | Issuer | Year\n\n"

        "━━━ CONTENT RULES (non-negotiable) ━━━\n"
        "1. NEVER add a job, project, certification, or skill not in the original resume.\n"
        "2. Only include a section if the original has content for it.\n"
        "3. Keep every job's company name, title, and dates exactly as written in the original.\n"
        "4. Keep the same number of bullet points per job as the original — no additions, no removals.\n"
        "5. You MAY rephrase a bullet to use the job description's exact terminology "
        "if the underlying work is genuinely the same thing.\n"
        "6. If a keyword has no honest hook in the original experience, skip it.\n"
        "7. Do not invent metrics or percentages not already in the original.\n\n"

        "━━━ OPTIMIZATION GOAL ━━━\n"
        f"Strengthen these already-matched keywords throughout:\n{matched}\n\n"
        f"Fix these ATS issues using existing content only:\n{issue_list}\n\n"
        f"Job Description (keyword reference):\n{job_text}\n\n"
        f"Original Resume (content source of truth):\n{resume_text}\n\n"
        "Output the full rewritten resume now in the markdown format above."
    )

    with claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def refine_resume(current_resume: str, user_request: str, original_resume: str = ""):
    """
    Apply the user's requested change to the current optimized resume.

    Supports two modes driven by the user's request:
    - General edit: make ONLY the specific change asked for
    - Style restore: if the user asks to 'preserve original style' or 'use original format',
      reformat the resume back to match the original's layout while keeping all
      ATS-optimized wording from the current version

    Yields text chunks for streaming display.
    """
    style_keywords = (
        "original style", "original format", "original template",
        "original structure", "original layout", "preserve style",
        "keep original", "revert style", "my original"
    )
    is_style_request = any(kw in user_request.lower() for kw in style_keywords)

    if is_style_request and original_resume:
        prompt = (
            "You are a professional resume editor.\n\n"
            "The user wants their resume reformatted to match the original's visual style "
            "and structure, while keeping the ATS-optimized wording from the current version.\n\n"
            "Instructions:\n"
            "1. Use the ORIGINAL RESUME as your formatting template "
            "(section names, section order, heading style, bullet style, date formats, etc.).\n"
            "2. Use the CURRENT RESUME as your content source "
            "(keep all the improved, ATS-optimized wording — do not revert the wording).\n"
            "3. Where the original had a section the current version restructured, "
            "restore the original section name and order.\n"
            "4. Return the COMPLETE resume.\n\n"
            f"Original Resume (use as formatting template):\n{original_resume}\n\n"
            f"Current Resume (use as content/wording source):\n{current_resume}\n\n"
            "Output the reformatted resume now."
        )
    else:
        prompt = (
            "You are a professional resume editor.\n\n"
            "Apply ONLY the specific change the user asked for — do not touch anything else.\n\n"
            "Rules:\n"
            "1. Make exactly the change requested, nothing more.\n"
            "2. Keep ALL existing ATS keywords and phrases intact.\n"
            "3. Never add jobs, skills, or projects not already in the resume.\n"
            "4. Return the COMPLETE updated resume in markdown format.\n\n"
            f"Current Resume:\n{current_resume}\n\n"
            f"User's request: {user_request}\n\n"
            "Return the full updated resume now."
        )

    with claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def analyze_skill_gaps(resume_text: str, job_text: str, ats: dict) -> dict:
    """
    Identify skills the candidate is genuinely missing and suggest projects to build them.

    Uses Claude Tool Use for structured output. Returns a dict with:
      - current_score
      - projected_score_if_all_gaps_filled
      - gaps: list of {skill, why_it_matters, suggested_project, time_to_learn, score_points_gained}
    """
    missing      = ats.get("missing_keywords", [])
    nice_missing = ats.get("nice_to_have_missing", [])
    current      = ats.get("ats_score", 0)

    if not missing and not nice_missing:
        # Nothing to suggest — return empty plan
        return {"current_score": current, "projected_score_if_all_gaps_filled": current, "gaps": []}

    prompt = (
        "You are a technical career coach.\n\n"
        "A candidate has applied for the job below. Their resume is also provided. "
        "Based ONLY on what is absent from their resume, identify the skill gaps "
        "that are most important to close for this specific role.\n\n"
        "For each gap, recommend one concrete project they can build using their "
        "existing background as a foundation — something realistic they could start today.\n\n"
        "Do NOT recommend skills the candidate already has. "
        "Do NOT pad the list — only include gaps that genuinely matter for this role.\n\n"
        f"Current ATS score: {current}/100\n"
        f"Required keywords missing from resume: {', '.join(missing)}\n"
        f"Nice-to-have keywords missing: {', '.join(nice_missing)}\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Candidate Resume:\n{resume_text}"
    )

    response = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2000,
        tools=[SKILL_GAP_TOOL],
        tool_choice={"type": "tool", "name": "skill_gap_plan"},
        messages=[{"role": "user", "content": prompt}],
    )
    result = response.content[0].input
    # Coerce score_points_gained to int for every gap (Claude sometimes returns strings)
    for gap in result.get("gaps", []):
        gap["score_points_gained"] = int(gap.get("score_points_gained") or 0)
    return result


def stream_cover_letter(resume_text: str, job_text: str, ats: dict):
    """
    Write a personalized cover letter using OpenAI with streaming.
    Yields text chunks for live streaming display.
    """
    strengths = ", ".join(ats.get("strengths", [])[:3])

    prompt = (
        "Write a compelling, authentic cover letter for this job application.\n\n"
        f"Job Description:\n{job_text[:3000]}\n\n"
        f"Candidate Resume:\n{resume_text[:3000]}\n\n"
        f"Resume strengths to highlight: {strengths}\n\n"
        "Writing rules:\n"
        "- 3 concise paragraphs\n"
        "- Opening: a specific hook about this company's mission — "
        'NOT "I am writing to express my interest"\n'
        "- Middle: connect exactly 2 resume achievements to the job's pain points\n"
        "- Closing: confident and direct, matching the company's tone\n"
        "- Mirror the job description's exact keywords and phrases\n"
        '- Banned phrases: "great fit", "passionate about", "to whom it may concern"\n'
        "- Use markdown formatting"
    )

    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You write authentic cover letters that sound like real, thoughtful people. "
                    "You never use templates or generic phrases. "
                    "You always open with something specific about the company."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — PAGE SETUP & STYLING
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="\U0001f4bc",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* Hide Streamlit chrome */
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem; max-width: 1300px; }

  /* ── Top bar ── */
  .topbar {
    display: flex; align-items: baseline; gap: 0.75rem;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 1rem; margin-bottom: 2rem;
  }
  .topbar-title { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 0; }
  .topbar-sub   { font-size: 0.88rem; color: #6b7280; margin: 0; }

  /* ── Input section labels ── */
  .panel-label {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: #9ca3af; margin-bottom: 0.35rem;
  }

  /* ── ATS score circle card ── */
  .score-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    display: flex; align-items: center; gap: 2rem;
  }
  .score-circle {
    width: 90px; height: 90px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-weight: 700; flex-shrink: 0; border: 5px solid;
  }
  .score-number { font-size: 1.8rem; line-height: 1; }
  .score-pct    { font-size: 0.65rem; letter-spacing: 0.05em; margin-top: 2px; }
  .score-red    { border-color: #ef4444; color: #ef4444; }
  .score-amber  { border-color: #f59e0b; color: #f59e0b; }
  .score-green  { border-color: #10b981; color: #10b981; }
  .score-meta h3 { margin: 0 0 0.2rem; font-size: 1rem; color: #111827; }
  .score-meta p  { margin: 0; font-size: 0.85rem; color: #6b7280; }

  /* ── Keyword chips ── */
  .chip {
    display: inline-block; border-radius: 4px;
    padding: 3px 10px; font-size: 0.78rem;
    font-weight: 500; margin: 2px 3px 2px 0;
  }
  .chip-match   { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
  .chip-missing { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
  .chip-nice    { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

  /* ── Section sub-headers in output ── */
  .sub-header {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #9ca3af;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.8rem; margin: 1rem 0 0.5rem;
  }

  /* ── Empty output state ── */
  .empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 520px; border: 2px dashed #e5e7eb;
    border-radius: 10px; color: #d1d5db; gap: 0.5rem;
  }
  .empty-icon { font-size: 2rem; }
  .empty-text { font-size: 0.88rem; }

  /* ── Primary button ── */
  div[data-testid="stButton"] > button[kind="primary"] {
    background: #2563eb; border: none;
    font-weight: 600; letter-spacing: 0.02em;
    padding: 0.55rem 0; border-radius: 6px;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #1d4ed8;
  }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — UI: HEADER
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown(
    '<div class="topbar">'
    '<span class="topbar-title">\U0001f4bc Job Application Assistant</span>'
    '<span class="topbar-sub">'
    "Upload your resume and a job description — get your ATS score, "
    "an optimized resume, and a cover letter."
    "</span></div>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — UI: LEFT PANEL (Inputs)
# ═══════════════════════════════════════════════════════════════════════════════

with left_col:

    # ── Resume input ──────────────────────────────────────────────────────────
    st.markdown('<p class="panel-label">Your Resume</p>', unsafe_allow_html=True)
    resume_mode = st.radio(
        "resume_input_mode",
        ["Upload PDF", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    resume_text = ""
    if resume_mode == "Upload PDF":
        uploaded = st.file_uploader(
            "resume_upload",
            type=["pdf", "txt"],
            label_visibility="collapsed",
        )
        if uploaded:
            raw_bytes = uploaded.read()
            resume_text = (
                read_pdf(raw_bytes)
                if uploaded.type == "application/pdf"
                else raw_bytes.decode("utf-8", errors="ignore")
            )
            st.success(f"Loaded — {len(resume_text):,} characters extracted.")
    else:
        resume_text = st.text_area(
            "resume_paste",
            placeholder="Paste your full resume here...",
            height=220,
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Job description input ─────────────────────────────────────────────────
    st.markdown('<p class="panel-label">Job Description</p>', unsafe_allow_html=True)
    job_mode = st.radio(
        "job_input_mode",
        ["Paste URL", "Paste text"],
        horizontal=True,
        label_visibility="collapsed",
    )

    job_text = ""
    if job_mode == "Paste URL":
        job_url = st.text_input(
            "job_url",
            placeholder="https://jobs.lever.co/company/job-id",
            label_visibility="collapsed",
        )
        if job_url and job_url.startswith("http"):
            with st.spinner("Fetching job posting..."):
                scraped, err = scrape_job_url(job_url)
            if scraped:
                job_text = scraped
                st.success(f"Fetched — {len(job_text):,} characters.")
            else:
                st.warning(f"Could not fetch ({err}). Switch to Paste text.")
    else:
        job_text = st.text_area(
            "job_paste",
            placeholder="Paste the full job description here...",
            height=220,
            label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Generate button ───────────────────────────────────────────────────────
    ready    = bool(resume_text.strip()) and bool(job_text.strip())
    generate = st.button(
        "Analyze & Optimize Resume",
        type="primary",
        use_container_width=True,
        disabled=not ready,
    )
    if not ready:
        st.caption("Add your resume and the job description to enable this button.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — UI: RIGHT PANEL (Output)
#
# Session-state design: results persist across Streamlit reruns (chat input,
# tab clicks, etc. all trigger reruns). Each expensive AI call is cached once
# in session_state so it never re-runs unless the user clicks "Analyze" again.
#
# Keys used:
#   results_ready   - bool: True once the user has clicked Analyze
#   saved_resume    - the resume text that was analyzed
#   saved_job       - the job text that was analyzed
#   ats             - dict: structured ATS analysis result
#   optimized_text  - str: current resume text (updated by chat refinements)
#   ats_after       - dict: ATS score after the initial rewrite (fixed; not updated by chat)
#   gap_plan        - dict: skill gap plan result
#   resume_chat     - list of {role, content} chat messages
#   cover_letter    - str: generated cover letter
# ═══════════════════════════════════════════════════════════════════════════════

with right_col:

    # ── When "Analyze" is clicked: save inputs, clear all previous results ────
    if generate:
        send_pushover(
            title="Job Application Assistant",
            message="Someone just used the ATS optimizer!",
        )
        st.session_state["results_ready"]  = True
        st.session_state["saved_resume"]   = resume_text
        st.session_state["saved_job"]      = job_text
        # Clear cached results so everything re-runs fresh with new inputs
        for key in ["ats", "optimized_text", "ats_after", "gap_plan", "cover_letter"]:
            st.session_state.pop(key, None)
        st.session_state["resume_chat"] = []

    # ── Empty state — no analysis run yet ────────────────────────────────────
    if not st.session_state.get("results_ready"):
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">\U0001f4ca</div>'
            '<div class="empty-text">'
            "Your ATS score and optimized resume will appear here."
            "</div></div>",
            unsafe_allow_html=True,
        )

    else:
        # Pull saved inputs (widget values may be empty on chat-triggered reruns)
        r_text = st.session_state["saved_resume"]
        j_text = st.session_state["saved_job"]

        tab_ats, tab_resume, tab_gaps, tab_letter = st.tabs(
            ["ATS Score", "Optimized Resume", "Skill Gap Plan", "Cover Letter"]
        )

        # ── TAB 1: ATS Score ──────────────────────────────────────────────────
        with tab_ats:
            # Run only once; cached in session_state for all subsequent reruns
            if "ats" not in st.session_state:
                with st.spinner("Analyzing ATS compatibility..."):
                    try:
                        st.session_state["ats"] = analyze_ats(r_text, j_text)
                    except Exception as exc:
                        st.error(f"ATS analysis failed: {exc}")
                        st.stop()

            ats          = st.session_state["ats"]
            score        = ats.get("ats_score", 0)
            matched      = ats.get("matched_keywords", [])
            missing      = ats.get("missing_keywords", [])
            nice_missing = ats.get("nice_to_have_missing", [])

            # Score circle card
            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-circle {score_color(score)}">'
                f'<span class="score-number">{score}</span>'
                f'<span class="score-pct">/ 100</span>'
                f"</div>"
                f'<div class="score-meta">'
                f"<h3>{score_verdict(score)}</h3>"
                f"<p>{len(matched)} of {len(matched) + len(missing)} required keywords matched</p>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
            st.progress(score / 100)

            st.markdown('<div class="sub-header">Matched Keywords</div>', unsafe_allow_html=True)
            if matched:
                st.markdown(
                    "".join(f'<span class="chip chip-match">&#10003; {k}</span>' for k in matched),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None found.")

            st.markdown('<div class="sub-header">Missing Required Keywords</div>', unsafe_allow_html=True)
            if missing:
                st.markdown(
                    "".join(f'<span class="chip chip-missing">&#10007; {k}</span>' for k in missing),
                    unsafe_allow_html=True,
                )
            else:
                st.success("No required keywords missing — great match!")

            if nice_missing:
                st.markdown('<div class="sub-header">Nice-to-Have Not in Resume</div>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="chip chip-nice">{k}</span>' for k in nice_missing),
                    unsafe_allow_html=True,
                )

            col_issues, col_strengths = st.columns(2)
            with col_issues:
                if ats.get("issues"):
                    st.markdown('<div class="sub-header">ATS Issues to Fix</div>', unsafe_allow_html=True)
                    for issue in ats["issues"]:
                        st.markdown(f"- {issue}")
            with col_strengths:
                if ats.get("strengths"):
                    st.markdown('<div class="sub-header">Strengths</div>', unsafe_allow_html=True)
                    for strength in ats["strengths"]:
                        st.markdown(f"- {strength}")

        # ── TAB 2: Optimized Resume ────────────────────────────────────────────
        with tab_resume:
            if "ats" not in st.session_state:
                st.info("Open the ATS Score tab first to run the analysis.")
            else:
                ats          = st.session_state["ats"]
                score_before = ats.get("ats_score", 0)

                st.info(
                    "Rewritten using **only your existing experience** — no invented skills or projects. "
                    "Use the chat below to refine anything before downloading."
                )

                # ── Resume display area ──────────────────────────────────────
                # st.empty() lets us update the displayed text when chat refines the resume
                resume_display = st.empty()

                if "optimized_text" not in st.session_state:
                    # First load: stream the rewrite and cache it
                    optimized_text = ""
                    for chunk in stream_optimized_resume(r_text, j_text, ats):
                        optimized_text += chunk
                        resume_display.markdown(optimized_text)
                    st.session_state["optimized_text"] = optimized_text
                else:
                    # Subsequent reruns (e.g. after a chat message): show cached version
                    resume_display.markdown(st.session_state["optimized_text"])

                # ── PDF download ─────────────────────────────────────────────
                st.divider()
                try:
                    pdf_bytes = build_pdf(st.session_state["optimized_text"])
                    st.download_button(
                        label="Download Resume as PDF",
                        data=pdf_bytes,
                        file_name="optimized_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
                except Exception as pdf_err:
                    st.warning(f"PDF generation failed: {pdf_err}")

                # ── Before vs After ATS re-check (run once, cached) ──────────
                st.divider()
                st.markdown("#### ATS Score: Before vs After Rewrite")

                if "ats_after" not in st.session_state:
                    with st.spinner("Checking score on the optimized resume..."):
                        try:
                            st.session_state["ats_after"] = analyze_ats(
                                st.session_state["optimized_text"], j_text
                            )
                        except Exception as exc:
                            st.warning(f"Could not re-check ATS score: {exc}")

                if "ats_after" in st.session_state:
                    ats_after   = st.session_state["ats_after"]
                    score_after = ats_after.get("ats_score", 0)
                    delta       = score_after - score_before
                    delta_color = "#10b981" if delta > 0 else "#6b7280"
                    delta_str   = f"+{delta}" if delta > 0 else str(delta)
                    c_before    = score_color(score_before)
                    c_after     = score_color(score_after)

                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:2rem;padding:1.2rem;'
                        f'background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;">'
                        f"<div>"
                        f'<div style="font-size:0.72rem;text-transform:uppercase;'
                        f'letter-spacing:.06em;color:#9ca3af;margin-bottom:4px">Before</div>'
                        f'<div class="score-circle {c_before}" style="width:72px;height:72px;'
                        f'border-radius:50%;display:flex;flex-direction:column;align-items:center;'
                        f'justify-content:center;font-weight:700;border:4px solid;">'
                        f'<span style="font-size:1.5rem;line-height:1">{score_before}</span>'
                        f'<span style="font-size:0.6rem">/ 100</span></div></div>'
                        f'<div style="font-size:1.8rem;color:#d1d5db">&#8594;</div>'
                        f"<div>"
                        f'<div style="font-size:0.72rem;text-transform:uppercase;'
                        f'letter-spacing:.06em;color:#9ca3af;margin-bottom:4px">After rewrite</div>'
                        f'<div class="score-circle {c_after}" style="width:72px;height:72px;'
                        f'border-radius:50%;display:flex;flex-direction:column;align-items:center;'
                        f'justify-content:center;font-weight:700;border:4px solid;">'
                        f'<span style="font-size:1.5rem;line-height:1">{score_after}</span>'
                        f'<span style="font-size:0.6rem">/ 100</span></div></div>'
                        f'<div style="font-size:1.3rem;font-weight:700;color:{delta_color}">'
                        f"&#9650; {delta_str}</div>"
                        f'<div style="color:#374151;font-size:0.9rem">'
                        f"<strong>{score_verdict(score_after)}</strong><br>"
                        f'<span style="color:#6b7280;font-size:0.82rem">'
                        f"{len(ats_after.get('matched_keywords', []))} of "
                        f"{len(ats_after.get('matched_keywords', [])) + len(ats_after.get('missing_keywords', []))} "
                        f"keywords matched</span></div></div>",
                        unsafe_allow_html=True,
                    )

                    # ── ATS pass / fail verdict ───────────────────────────────
                    if score_after >= 75:
                        st.success(
                            "**Likely to pass ATS screening.** "
                            "Most automated filters look for a 70-75% keyword match. "
                            "This resume clears that bar — a recruiter should see it."
                        )
                    elif score_after >= 60:
                        st.warning(
                            "**Borderline — may pass some ATS systems.** "
                            "A score in this range gets through on lenient filters but risks "
                            "rejection on stricter ones. Check the Skill Gap Plan tab to push it higher."
                        )
                    else:
                        st.error(
                            "**At risk of being filtered out.** "
                            "Scores below 60 are commonly auto-rejected before a human sees the resume. "
                            "Review the Skill Gap Plan tab for the fastest ways to improve."
                        )

                # ── Chat console ─────────────────────────────────────────────
                # The user can ask to adjust anything; the resume display above
                # and the PDF download button update automatically after each turn.
                st.divider()
                st.markdown("#### Ask to refine your resume")
                st.caption(
                    "Make any adjustments before downloading — shorten a section, "
                    "change tone, reorder bullets, remove a skill, etc."
                )

                # Show chat history from session state
                for msg in st.session_state.get("resume_chat", []):
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])

                # Handle new chat message
                if user_msg := st.chat_input(
                    "e.g. Make the summary 2 sentences. Remove Docker. Add more metrics to the first job."
                ):
                    # Show the user's message immediately
                    with st.chat_message("user"):
                        st.markdown(user_msg)

                    # Stream the refined resume into the main display area
                    with st.chat_message("assistant"):
                        status = st.empty()
                        status.markdown("Refining your resume...")

                        updated = ""
                        for chunk in refine_resume(
                            st.session_state["optimized_text"],
                            user_msg,
                            original_resume=st.session_state.get("saved_resume", ""),
                        ):
                            updated += chunk
                            resume_display.markdown(updated)  # live update at the top

                        status.markdown(
                            "Done! The resume above has been updated. "
                            "The download button now reflects the latest version."
                        )

                    # Save updated resume and chat history, then rerun so the
                    # download button re-renders with the new PDF content
                    st.session_state["optimized_text"] = updated
                    st.session_state["resume_chat"].append(
                        {"role": "user", "content": user_msg}
                    )
                    st.session_state["resume_chat"].append(
                        {"role": "assistant", "content":
                            "Done! The resume has been updated. "
                            "Download the latest version when you're happy with it."}
                    )
                    st.rerun()

        # ── TAB 3: Skill Gap Plan ─────────────────────────────────────────────
        with tab_gaps:
            if "ats" not in st.session_state:
                st.info("Open the ATS Score tab first to run the analysis.")
            else:
                if "gap_plan" not in st.session_state:
                    with st.spinner("Identifying skill gaps and building your learning plan..."):
                        try:
                            st.session_state["gap_plan"] = analyze_skill_gaps(
                                r_text, j_text, st.session_state["ats"]
                            )
                        except Exception as exc:
                            st.error(f"Skill gap analysis failed: {exc}")

                if "gap_plan" in st.session_state:
                    gap_plan = st.session_state["gap_plan"]
                    gaps     = gap_plan.get("gaps", [])

                    if not gaps:
                        st.success(
                            "No significant skill gaps found. "
                            "Your background already covers the core requirements for this role."
                        )
                    else:
                        current_sc     = st.session_state["ats"].get("ats_score", 0)
                        raw_gain       = sum(g.get("score_points_gained", 0) for g in gaps)
                        projected_sc   = min(current_sc + raw_gain, 100)
                        display_gain   = projected_sc - current_sc

                        st.markdown(
                            f'<div style="background:#fffbeb;border:1px solid #fde68a;'
                            f'border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.2rem;">'
                            f'<div style="font-weight:700;color:#92400e;font-size:0.95rem;">'
                            f'Current score: <strong>{current_sc}</strong> &nbsp;&#8594;&nbsp; '
                            f'Projected score if all skills added: '
                            f'<span style="font-size:1.2rem">~{projected_sc} / 100</span>'
                            f'</div>'
                            f'<div style="font-size:0.82rem;color:#78350f;margin-top:4px;">'
                            f'Potential gain of <strong>+{display_gain} pts</strong>. '
                            f'Per-skill estimates below are based on each skill\'s weight in the job description.'
                            f'</div></div>',
                            unsafe_allow_html=True,
                        )

                        for i, gap in enumerate(gaps):
                            skill    = gap.get("skill", "")
                            why      = gap.get("why_it_matters", "")
                            project  = gap.get("suggested_project", "")
                            time_est = gap.get("time_to_learn", "")
                            pts      = gap.get("score_points_gained", 0)

                            with st.expander(
                                f"**{skill}**  —  estimated +{pts} pts  |  {time_est}",
                                expanded=(i == 0),
                            ):
                                st.markdown("**Why this role needs it**")
                                st.markdown(why)
                                st.markdown("**Project to build this skill**")
                                st.markdown(
                                    f'<div style="background:#f0fdf4;border-left:3px solid #10b981;'
                                    f'padding:0.8rem 1rem;border-radius:0 6px 6px 0;'
                                    f'font-size:0.9rem;color:#111827;">{project}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(f"Estimated time: {time_est}  |  Estimated ATS gain: +{pts} pts")

                        st.divider()
                        st.markdown(
                            '<div style="background:#f0fdf4;border:1px solid #bbf7d0;'
                            'border-radius:10px;padding:1rem 1.4rem;font-size:0.9rem;color:#166534;">'
                            '<strong>Done with one of these projects?</strong><br>'
                            'Add it to your resume, come back, and paste the updated version '
                            'to get a fresh ATS score and see exactly how much it improved.'
                            '</div>',
                            unsafe_allow_html=True,
                        )

        # ── TAB 4: Cover Letter ────────────────────────────────────────────────
        with tab_letter:
            if "ats" not in st.session_state:
                st.info("Open the ATS Score tab first to run the analysis.")
            else:
                if "cover_letter" not in st.session_state:
                    letter_placeholder = st.empty()
                    cover_letter_text  = ""
                    for chunk in stream_cover_letter(r_text, j_text, st.session_state["ats"]):
                        cover_letter_text += chunk
                        letter_placeholder.markdown(cover_letter_text)
                    st.session_state["cover_letter"] = cover_letter_text
                else:
                    st.markdown(st.session_state["cover_letter"])

                with st.expander("Copy plain text"):
                    st.text_area(
                        "letter_copy",
                        value=st.session_state["cover_letter"],
                        height=350,
                        label_visibility="collapsed",
                    )
