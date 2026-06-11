import io, re, unicodedata
from fpdf import FPDF

sample = """# John Smith
Senior Machine Learning Engineer | john@email.com

## Work Experience

### TechVision AI -- Senior ML Engineer (2022-Present)
- Deployed fraud detection API serving 10,000 requests/day with 99.9% uptime
- Built RAG pipeline (ChromaDB + Claude) cutting research time by 60%
- Led monolith to microservices migration, deploy time from 3hr to 15min

## Skills
Python, PyTorch, AWS, Docker, Kubernetes, PostgreSQL, Redis, LangChain

## Education
BS Computer Science, State University, 2020
"""

def to_ascii(text):
    return unicodedata.normalize("NFKD", text).encode("ascii", errors="ignore").decode("ascii")

def clean(line):
    line = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
    line = re.sub(r"\*(.+?)\*",     r"\1", line)
    line = re.sub(r"_[^_\n]*_",     "",    line)
    return to_ascii(line.strip())

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=20)
pdf.add_page()
pdf.set_margins(20, 20, 20)

def put_cell(text, style="B", size=11, color=(17,24,39), h=7):
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.set_x(pdf.l_margin)
    pdf.cell(0, h, text, new_x="LMARGIN", new_y="NEXT")

def put_text(text, style="", size=10, color=(17,24,39)):
    pdf.set_font("Helvetica", style, size)
    pdf.set_text_color(*color)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 5, text)

for raw_line in sample.split("\n"):
    line = raw_line.strip()
    if not line:
        pdf.ln(3)
    elif line.startswith("# "):
        pdf.ln(2); put_cell(clean(line[2:]), size=15, h=9)
    elif line.startswith("## "):
        pdf.ln(4); put_cell(clean(line[3:]).upper(), color=(37,99,235))
        pdf.set_draw_color(229,231,235)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2)
    elif line.startswith("### "):
        pdf.ln(2); put_cell(clean(line[4:]), size=10, color=(55,65,81), h=6)
    elif line.startswith("- "):
        put_text("  - " + clean(line[2:]))
    else:
        put_text(clean(line))

buf = io.BytesIO()
buf.write(pdf.output())
data = buf.getvalue()

with open("test_output.pdf", "wb") as f:
    f.write(data)

print(f"OK: {len(data):,} bytes -> test_output.pdf")
