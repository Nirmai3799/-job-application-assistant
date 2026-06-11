"""
AI Job Application Assistant -- ATS Optimizer
Run:  streamlit run app.py
"""

import os
import io
import json
import re
import unicodedata
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import streamlit as st
from openai import OpenAI
import anthropic
import PyPDF2
from fpdf import FPDF

# -- Page config ---------------------------------------------------------------
st.set_page_config(
    page_title="Job Application Assistant",
    page_icon="\U0001f4bc",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- CSS -----------------------------------------------------------------------
st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding: 2rem 2.5rem; max-width: 1300px; }

  .topbar {
    display: flex; align-items: baseline; gap: 0.75rem;
    border-bottom: 1px solid #e5e7eb;
    padding-bottom: 1rem; margin-bottom: 2rem;
  }
  .topbar-title { font-size: 1.25rem; font-weight: 700; color: #111827; margin: 0; }
  .topbar-sub   { font-size: 0.88rem; color: #6b7280; margin: 0; }

  .panel-label {
    font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.07em; text-transform: uppercase;
    color: #9ca3af; margin-bottom: 0.35rem;
  }

  .score-card {
    background: #f9fafb; border: 1px solid #e5e7eb;
    border-radius: 10px; padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem; display: flex;
    align-items: center; gap: 2rem;
  }
  .score-circle {
    width: 90px; height: 90px; border-radius: 50%;
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    font-weight: 700; flex-shrink: 0; border: 5px solid;
  }
  .score-number { font-size: 1.8rem; line-height: 1; }
  .score-label  { font-size: 0.65rem; text-transform: uppercase;
                  letter-spacing: 0.05em; margin-top: 2px; }
  .score-red    { border-color: #ef4444; color: #ef4444; }
  .score-amber  { border-color: #f59e0b; color: #f59e0b; }
  .score-green  { border-color: #10b981; color: #10b981; }
  .score-meta h3 { margin: 0 0 0.2rem; font-size: 1rem; color: #111827; }
  .score-meta p  { margin: 0; font-size: 0.85rem; color: #6b7280; }

  .chip {
    display: inline-block; border-radius: 4px;
    padding: 3px 10px; font-size: 0.78rem;
    font-weight: 500; margin: 2px 3px 2px 0;
  }
  .chip-match   { background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; }
  .chip-missing { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
  .chip-nice    { background: #eff6ff; color: #1d4ed8; border: 1px solid #bfdbfe; }

  .section-divider {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.07em;
    text-transform: uppercase; color: #9ca3af;
    border-top: 1px solid #f3f4f6;
    padding-top: 0.8rem; margin: 1rem 0 0.5rem;
  }

  .empty-state {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    height: 520px; border: 2px dashed #e5e7eb;
    border-radius: 10px; color: #d1d5db; gap: 0.5rem;
  }
  .empty-icon { font-size: 2rem; }
  .empty-text { font-size: 0.88rem; }

  div[data-testid="stButton"] > button[kind="primary"] {
    background: #2563eb; border: none;
    font-weight: 600; letter-spacing: 0.02em;
    padding: 0.55rem 0; border-radius: 6px;
  }
  div[data-testid="stButton"] > button[kind="primary"]:hover { background: #1d4ed8; }
</style>
""", unsafe_allow_html=True)

# -- Clients -------------------------------------------------------------------
load_dotenv()


@st.cache_resource
def get_clients():
    return (
        anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")),
        OpenAI(api_key=os.getenv("OPENAI_API_KEY")),
    )


claude_client, openai_client = get_clients()
CLAUDE_MODEL = "claude-sonnet-4-6"
OPENAI_MODEL = "gpt-4.1-mini"


# -- Helpers -------------------------------------------------------------------
def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def fetch_job_from_url(url: str, max_chars: int = 8000) -> tuple:
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
            l.strip()
            for l in soup.get_text("\n", strip=True).split("\n")
            if len(l.strip()) > 5
        ]
        return "\n".join(lines)[:max_chars], None
    except Exception as exc:
        return None, str(exc)


def strip_fences(text: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())


def generate_pdf(markdown_text: str) -> bytes:
    """
    Convert resume markdown to PDF using Helvetica (fpdf2 built-in, no file I/O).
    unicodedata normalises any Unicode characters Claude outputs to their
    closest ASCII equivalent so Helvetica's latin-1 range is never exceeded.
    """

    def to_ascii(text: str) -> str:
        # Decompose accented chars (e.g. é -> e + combining accent), keep ASCII
        normalized = unicodedata.normalize("NFKD", text)
        return normalized.encode("ascii", errors="ignore").decode("ascii")

    def clean(line: str) -> str:
        line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)   # **bold** -> plain
        line = re.sub(r"\*(.+?)\*",     r"\1", line)   # *italic* -> plain
        line = re.sub(r"_[^_\n]*_",     "",    line)   # _Targets:..._ -> removed
        return to_ascii(line.strip())

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    def put_cell(text, style="B", size=11, color=(17, 24, 39), h=7):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.cell(0, h, text, new_x="LMARGIN", new_y="NEXT")

    def put_text(text, style="", size=10, color=(17, 24, 39)):
        pdf.set_font("Helvetica", style, size)
        pdf.set_text_color(*color)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, text)

    for raw_line in markdown_text.split("\n"):
        line = raw_line.strip()

        if not line:
            pdf.ln(3)

        elif line.startswith("# "):
            pdf.ln(2)
            put_cell(clean(line[2:]), size=15, h=9)

        elif line.startswith("## "):
            pdf.ln(4)
            put_cell(clean(line[3:]).upper(), color=(37, 99, 235))
            pdf.set_draw_color(229, 231, 235)
            pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
            pdf.ln(2)

        elif line.startswith("### "):
            pdf.ln(2)
            put_cell(clean(line[4:]), size=10, color=(55, 65, 81), h=6)

        elif line.startswith(("- ", "* ")):
            put_text("  - " + clean(line[2:]))

        else:
            put_text(clean(line))

    buf = io.BytesIO()
    buf.write(pdf.output())
    return buf.getvalue()


def score_color_class(score: int) -> str:
    if score >= 75:
        return "score-green"
    if score >= 50:
        return "score-amber"
    return "score-red"


def score_label(score: int) -> str:
    if score >= 75:
        return "Strong match"
    if score >= 50:
        return "Partial match - needs work"
    return "Low match - likely rejected"


# -- LLM calls -----------------------------------------------------------------
def run_ats_analysis(resume_text: str, job_text: str) -> dict:
    prompt = (
        "You are an ATS (Applicant Tracking System) expert.\n\n"
        "Analyze how well this resume matches the job description.\n\n"
        "Return ONLY a JSON object - no fences, no explanation:\n"
        "{\n"
        '  "ats_score": <integer 0-100>,\n'
        '  "matched_keywords": ["keywords present in both resume and job"],\n'
        '  "missing_keywords": ["required keywords completely absent from resume"],\n'
        '  "nice_to_have_missing": ["bonus skills from job not in resume"],\n'
        '  "issues": ["specific ATS red flags: missing sections, weak phrasing, etc."],\n'
        '  "strengths": ["what the resume does well for this role"]\n'
        "}\n\n"
        "Scoring guide:\n"
        "- 90-100: Near-perfect keyword match, strong ATS pass\n"
        "- 75-89:  Good match, minor gaps\n"
        "- 50-74:  Partial match, several key terms missing\n"
        "- 0-49:   Poor match, likely auto-rejected\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Resume:\n{resume_text}"
    )
    r = claude_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1200,
        system="You are an ATS expert. Return only valid JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(strip_fences(r.content[0].text))


def stream_optimized_resume(resume_text: str, job_text: str, ats_report: dict):
    missing      = ", ".join(ats_report.get("missing_keywords", []))
    nice_missing = ", ".join(ats_report.get("nice_to_have_missing", []))
    issues       = "\n".join(f"- {i}" for i in ats_report.get("issues", []))

    prompt = (
        "You are an expert resume writer specializing in ATS optimization.\n\n"
        "Rewrite the candidate's resume to maximize ATS score for this specific job.\n\n"
        "Rules:\n"
        "1. Incorporate EVERY missing keyword naturally - no keyword stuffing\n"
        "2. Use exact terminology from the job description (ATS does literal matching)\n"
        "3. Keep standard section headers: Summary, Work Experience, Skills, Education, Projects\n"
        "4. Quantify every achievement with numbers\n"
        "5. Start each bullet with a strong action verb\n"
        f"6. Fix these ATS issues:\n{issues}\n"
        "7. Output clean markdown (no tables, no columns)\n\n"
        f"Must add these missing required keywords: {missing}\n"
        f"Should also include if possible: {nice_missing}\n\n"
        f"Job Description:\n{job_text}\n\n"
        f"Original Resume:\n{resume_text}\n\n"
        "Write the complete optimized resume now."
    )

    with claude_client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=2500,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        for chunk in stream.text_stream:
            yield chunk


def stream_cover_letter(resume_text: str, job_text: str, ats_report: dict):
    strengths = ", ".join(ats_report.get("strengths", [])[:3])
    prompt = (
        "Write a compelling, authentic cover letter for this job application.\n\n"
        f"Job Description:\n{job_text[:3000]}\n\n"
        f"Candidate Resume:\n{resume_text[:3000]}\n\n"
        f"Key strengths to highlight: {strengths}\n\n"
        "Instructions:\n"
        "- 3 focused paragraphs\n"
        '- Open with a specific hook about the company - NOT "I am writing to express my interest..."\n'
        "- Para 2: Connect 2 specific resume achievements to the job's core requirements\n"
        "- Para 3: Confident, direct close\n"
        "- Mirror the exact keywords and terminology from the job description\n"
        '- Avoid all filler: "great fit", "passionate", "to whom it may concern"\n'
        "- Markdown formatting"
    )
    stream = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You write authentic cover letters. Never use generic templates. Mirror the job's exact language.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    for chunk in stream:
        yield chunk.choices[0].delta.content or ""


# -- UI ------------------------------------------------------------------------
st.markdown(
    '<div class="topbar">'
    '<span class="topbar-title">\U0001f4bc Job Application Assistant</span>'
    '<span class="topbar-sub">Get your ATS score, an optimized resume, and a cover letter.</span>'
    "</div>",
    unsafe_allow_html=True,
)

left_col, right_col = st.columns([1, 1], gap="large")

# -- LEFT: Inputs --------------------------------------------------------------
with left_col:

    st.markdown('<p class="panel-label">Your Resume</p>', unsafe_allow_html=True)
    resume_mode = st.radio(
        "resume_mode", ["Upload PDF", "Paste text"],
        horizontal=True, label_visibility="collapsed"
    )

    resume_text = ""
    if resume_mode == "Upload PDF":
        uploaded = st.file_uploader("upload", type=["pdf", "txt"], label_visibility="collapsed")
        if uploaded:
            raw = uploaded.read()
            resume_text = (
                extract_pdf_text(raw)
                if uploaded.type == "application/pdf"
                else raw.decode("utf-8", errors="ignore")
            )
            st.success(f"Loaded - {len(resume_text):,} characters extracted.")
    else:
        resume_text = st.text_area(
            "paste_resume", placeholder="Paste your full resume here...",
            height=220, label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<p class="panel-label">Job Description</p>', unsafe_allow_html=True)
    job_mode = st.radio(
        "job_mode", ["Paste URL", "Paste text"],
        horizontal=True, label_visibility="collapsed"
    )

    job_text = ""
    if job_mode == "Paste URL":
        job_url = st.text_input(
            "job_url", placeholder="https://jobs.lever.co/...",
            label_visibility="collapsed"
        )
        if job_url and job_url.startswith("http"):
            with st.spinner("Fetching job posting..."):
                scraped, err = fetch_job_from_url(job_url)
            if scraped:
                job_text = scraped
                st.success(f"Fetched - {len(job_text):,} characters.")
            else:
                st.warning(f"Could not fetch ({err}). Switch to Paste text.")
    else:
        job_text = st.text_area(
            "paste_job", placeholder="Paste the full job description here...",
            height=220, label_visibility="collapsed"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    ready = bool(resume_text.strip()) and bool(job_text.strip())
    generate = st.button(
        "Analyze & Optimize Resume", type="primary",
        use_container_width=True, disabled=not ready
    )
    if not ready:
        st.caption("Add your resume and the job description to continue.")


# -- RIGHT: Output -------------------------------------------------------------
with right_col:

    if not generate:
        st.markdown(
            '<div class="empty-state">'
            '<div class="empty-icon">\U0001f4ca</div>'
            '<div class="empty-text">Your ATS score and optimized resume will appear here.</div>'
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        tab_ats, tab_resume, tab_letter = st.tabs(
            ["ATS Score", "Optimized Resume", "Cover Letter"]
        )

        # ATS Score tab --------------------------------------------------------
        with tab_ats:
            with st.spinner("Analyzing ATS compatibility..."):
                try:
                    ats = run_ats_analysis(resume_text, job_text)
                    st.session_state["ats"]         = ats
                    st.session_state["resume_text"] = resume_text
                    st.session_state["job_text"]    = job_text
                except Exception as exc:
                    st.error(f"Analysis failed: {exc}")
                    st.stop()

            score        = ats.get("ats_score", 0)
            color        = score_color_class(score)
            verdict      = score_label(score)
            matched      = ats.get("matched_keywords", [])
            missing      = ats.get("missing_keywords", [])
            nice_missing = ats.get("nice_to_have_missing", [])

            st.markdown(
                f'<div class="score-card">'
                f'<div class="score-circle {color}">'
                f'<span class="score-number">{score}</span>'
                f'<span class="score-label">/ 100</span>'
                f"</div>"
                f'<div class="score-meta">'
                f"<h3>{verdict}</h3>"
                f"<p>{len(matched)} of {len(matched) + len(missing)} required keywords matched</p>"
                f"</div></div>",
                unsafe_allow_html=True,
            )

            st.progress(score / 100)

            st.markdown('<div class="section-divider">Matched Keywords</div>', unsafe_allow_html=True)
            if matched:
                st.markdown(
                    "".join(f'<span class="chip chip-match">&#10003; {k}</span>' for k in matched),
                    unsafe_allow_html=True,
                )
            else:
                st.caption("None found.")

            st.markdown('<div class="section-divider">Missing Required Keywords</div>', unsafe_allow_html=True)
            if missing:
                st.markdown(
                    "".join(f'<span class="chip chip-missing">&#10007; {k}</span>' for k in missing),
                    unsafe_allow_html=True,
                )
            else:
                st.success("No required keywords missing!")

            if nice_missing:
                st.markdown('<div class="section-divider">Nice-to-Have Not in Resume</div>', unsafe_allow_html=True)
                st.markdown(
                    "".join(f'<span class="chip chip-nice">{k}</span>' for k in nice_missing),
                    unsafe_allow_html=True,
                )

            col_a, col_b = st.columns(2)
            with col_a:
                if ats.get("issues"):
                    st.markdown('<div class="section-divider">ATS Issues to Fix</div>', unsafe_allow_html=True)
                    for issue in ats["issues"]:
                        st.markdown(f"- {issue}")
            with col_b:
                if ats.get("strengths"):
                    st.markdown('<div class="section-divider">Strengths</div>', unsafe_allow_html=True)
                    for s in ats["strengths"]:
                        st.markdown(f"- {s}")

        # Optimized Resume tab -------------------------------------------------
        with tab_resume:
            if "ats" not in st.session_state:
                st.info("Run ATS analysis first.")
            else:
                score_before  = st.session_state["ats"].get("ats_score", 0)
                missing_count = len(st.session_state["ats"].get("missing_keywords", []))
                st.caption(f"Incorporating **{missing_count} missing keyword(s)** into your resume...")

                placeholder = st.empty()
                optimized   = ""
                for chunk in stream_optimized_resume(
                    st.session_state["resume_text"],
                    st.session_state["job_text"],
                    st.session_state["ats"],
                ):
                    optimized += chunk
                    placeholder.markdown(optimized)

                # PDF download
                st.divider()
                try:
                    pdf_bytes = generate_pdf(optimized)
                    st.download_button(
                        label="Download Optimized Resume as PDF",
                        data=pdf_bytes,
                        file_name="optimized_resume.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                        type="primary",
                    )
                except Exception as pdf_err:
                    st.warning(f"PDF generation failed: {pdf_err}")

                # Before vs After ATS score
                st.divider()
                st.markdown("#### ATS Score: Before vs After")
                with st.spinner("Re-checking ATS score on optimized resume..."):
                    try:
                        ats_after   = run_ats_analysis(optimized, st.session_state["job_text"])
                        score_after = ats_after.get("ats_score", 0)
                        delta       = score_after - score_before
                        color_b     = score_color_class(score_before)
                        color_a     = score_color_class(score_after)
                        delta_color = "#10b981" if delta > 0 else "#6b7280"
                        delta_sign  = f"+{delta}" if delta > 0 else str(delta)

                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:2rem;padding:1.2rem;'
                            f'background:#f9fafb;border:1px solid #e5e7eb;border-radius:10px;">'
                            f"<div>"
                            f'<div style="font-size:0.72rem;text-transform:uppercase;'
                            f'letter-spacing:.06em;color:#9ca3af;margin-bottom:4px">Before</div>'
                            f'<div class="score-circle {color_b}" style="width:72px;height:72px;'
                            f"border-radius:50%;display:flex;flex-direction:column;"
                            f'align-items:center;justify-content:center;font-weight:700;border:4px solid;">'
                            f'<span style="font-size:1.5rem;line-height:1">{score_before}</span>'
                            f'<span style="font-size:0.6rem">/ 100</span></div></div>'
                            f'<div style="font-size:1.8rem;color:#d1d5db">&#8594;</div>'
                            f"<div>"
                            f'<div style="font-size:0.72rem;text-transform:uppercase;'
                            f'letter-spacing:.06em;color:#9ca3af;margin-bottom:4px">After</div>'
                            f'<div class="score-circle {color_a}" style="width:72px;height:72px;'
                            f"border-radius:50%;display:flex;flex-direction:column;"
                            f'align-items:center;justify-content:center;font-weight:700;border:4px solid;">'
                            f'<span style="font-size:1.5rem;line-height:1">{score_after}</span>'
                            f'<span style="font-size:0.6rem">/ 100</span></div></div>'
                            f'<div style="font-size:1.4rem;font-weight:700;color:{delta_color}">'
                            f"&#9650; {delta_sign}</div>"
                            f'<div style="color:#374151;font-size:0.9rem">'
                            f"<strong>{score_label(score_after)}</strong><br>"
                            f'<span style="color:#6b7280;font-size:0.82rem">'
                            f"{len(ats_after.get('matched_keywords', []))} of "
                            f"{len(ats_after.get('matched_keywords', [])) + len(ats_after.get('missing_keywords', []))} "
                            f"keywords matched</span></div></div>",
                            unsafe_allow_html=True,
                        )
                    except Exception as e:
                        st.warning(f"Could not re-check ATS score: {e}")

        # Cover Letter tab -----------------------------------------------------
        with tab_letter:
            if "ats" not in st.session_state:
                st.info("Run ATS analysis first.")
            else:
                placeholder2 = st.empty()
                letter = ""
                for chunk in stream_cover_letter(
                    st.session_state["resume_text"],
                    st.session_state["job_text"],
                    st.session_state["ats"],
                ):
                    letter += chunk
                    placeholder2.markdown(letter)

                with st.expander("Copy plain text"):
                    st.text_area("letter_copy", value=letter, height=350,
                                 label_visibility="collapsed")
