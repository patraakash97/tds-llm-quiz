import re
import io
import json
import base64
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse
from io import BytesIO

import requests
import pdfplumber
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans  # unused, kept for possible extension


# ============================================================
# 1. Utility helpers
# ============================================================

def origin_from_url(url: str) -> Optional[str]:
    """Return 'scheme://host' from a full URL."""
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else None
    except Exception:
        return None


def extract_question_and_submit_url(text: str, html: str, current_url: str):
    """
    Extract a question snippet and the /submit URL from page text/HTML.
    Works for demo page and similar formats.
    """
    submit_url = None

    # Explicit "Post your answer to ..."
    m = re.search(r"Post your answer to\s+(https?://\S+)", text)
    if m:
        submit_url = m.group(1).strip().rstrip(").,")

    # Fallback: any URL ending with /submit in HTML
    if submit_url is None:
        m2 = re.search(r"https?://[^\"'>\s]+/submit", html)
        if m2:
            submit_url = m2.group(0).strip().rstrip(").,")

    # Last fallback: origin + "/submit"
    if submit_url is None:
        origin = origin_from_url(current_url)
        if origin:
            submit_url = origin + "/submit"

    # Extract question text (starting from Qxxx if present)
    q_match = re.search(r"(Q\d+.*)", text)
    question = q_match.group(1).strip() if q_match else text[:200].strip()

    return question, submit_url


def extract_first_pdf_link(text: str, html: str):
    """Find first https://...pdf in HTML or text."""
    m = re.search(r'href="(https?://[^"]+\.pdf)"', html)
    if m:
        return m.group(1)

    m = re.search(r"(https?://\S+\.pdf)", text)
    if m:
        return m.group(1).rstrip(").,")

    return None


def extract_embedded_base64_blocks(html: str) -> str:
    """
    demo-scrape page has base64-encoded instructions inside JS template string.
    Decode those and return as extra text.
    """
    chunks = []
    # Look for things like: code = `U2NyYXBlIDxhIGhyZWY9Ii9kZW1v...`
    for m in re.finditer(r"`([A-Za-z0-9+/=\s]{40,})`", html):
        b64 = "".join(m.group(1).split())
        try:
            decoded = base64.b64decode(b64).decode("utf-8", errors="ignore")
            chunks.append(decoded)
        except Exception:
            continue
    return "\n".join(chunks)


# ============================================================
# 2. Built-in solvers
# ============================================================

def solve_pdf_sum_value_page2(pdf_bytes: bytes) -> float:
    """
    Example PDF helper: sum of ALL numeric cells on page 2.
    (Kept for future tasks.)
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) < 2:
            raise ValueError("PDF has less than 2 pages, cannot use page 2")
        page2 = pdf.pages[1]
        table = page2.extract_table()
        if not table:
            raise ValueError("No table found on PDF page 2")

        df = pd.DataFrame(table)
        flat = df.values.ravel()
        nums = pd.to_numeric(flat, errors="coerce")
        total = float(np.nansum(nums))
        return total


def solve_csv_basic(url: str, task: str):
    """Basic CSV tasks: mean/sum/max/min of a given column."""
    try:
        df = pd.read_csv(url)
    except Exception as e:
        return {"status": "csv-error", "error": f"Failed to load CSV: {str(e)}"}

    task_low = task.lower()

    def _col_from_pattern(pattern: str):
        m = re.search(pattern, task, re.I)
        return m.group(1) if m else None

    if "mean" in task_low:
        col = _col_from_pattern(r"mean.*?(\w+)")
        if col in df.columns:
            return float(df[col].mean())
        return {"status": "csv-error", "error": f"Column {col} not found"}

    if "sum" in task_low:
        col = _col_from_pattern(r"sum.*?(\w+)")
        if col in df.columns:
            return float(df[col].sum())
        return {"status": "csv-error", "error": f"Column {col} not found"}

    if "max" in task_low:
        col = _col_from_pattern(r"max.*?(\w+)")
        if col in df.columns:
            return float(df[col].max())
        return {"status": "csv-error", "error": f"Column {col} not found"}

    if "min" in task_low:
        col = _col_from_pattern(r"min.*?(\w+)")
        if col in df.columns:
            return float(df[col].min())
        return {"status": "csv-error", "error": f"Column {col} not found"}

    return df.to_dict(orient="records")


def solve_json_basic(url: str, task: str):
    """Basic JSON tasks like count or max(field)."""
    try:
        data = requests.get(url, timeout=20).json()
    except Exception as e:
        return {"status": "json-error", "error": f"Failed to parse JSON: {str(e)}"}

    task_low = task.lower()

    if isinstance(data, list):
        if "count" in task_low:
            return len(data)

        if "max" in task_low:
            m = re.search(r"max.*?(\w+)", task, re.I)
            field = m.group(1) if m else None
            if not field:
                return {"status": "json-error", "error": "Could not infer field for max"}
            try:
                return max(item[field] for item in data if field in item)
            except Exception as e:
                return {"status": "json-error", "error": f"Failed max calc: {e}"}

    return data


def parse_html_table(html: str):
    """
    Parse first HTML table from raw HTML using pandas.read_html.
    If this gives issues on some pages, we just return None.
    """
    try:
        tables = pd.read_html(html)
        if not tables:
            return None
        return tables[0]
    except Exception:
        return None


def solve_image_ocr(base64_img: str) -> str:
    """OCR disabled (no EasyOCR dependency on server)."""
    return "OCR-not-available"


def generate_plot_base64(df, x, y) -> str:
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y])
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 3. Per-page quiz solver (pure requests, no browser)
# ============================================================

def solve_single_quiz(question: str, text: str, html: str, current_url: str, email: str):
    """
    Decide how to answer one quiz page, based on question + content.
    """
    q_lower = question.lower()
    text_lower = text.lower()

    print("\n=== SOLVING QUESTION SNIPPET ===")
    print(question[:300], "...\n")

    # -------- DEMO TYPE 1: "anything you want" --------
    if "anything you want" in text_lower:
        return "hello-from-akash"

    # --------------------------------------------------
    # DEMO SCRAPE QUESTION (handles base64 JS + /demo-scrape-data)
    # --------------------------------------------------

    # Decode any embedded base64 instructions from JS
    decoded_extra = extract_embedded_base64_blocks(html)
    combined = text + "\n" + decoded_extra

    # Find /demo-scrape-data?email=... pattern
    m = re.search(r"(/demo-scrape-data[^\"'<\s]*)", combined)
    if m:
        relative = m.group(1).rstrip('">) ')
        # replace $EMAIL if present
        relative = relative.replace("$EMAIL", email)
        origin = origin_from_url(current_url)
        url = origin + relative if origin else relative
        print(f"[DEBUG] Scraping relative URL (auto): {url}")

        try:
            resp = requests.get(url, timeout=20)
            # demo-scrape-data returns plain text secret
            secret = resp.text.strip()
            if not secret:
                # fallback: try JSON and pick first string field
                try:
                    data = resp.json()
                    if isinstance(data, dict):
                        for v in data.values():
                            if isinstance(v, str):
                                secret = v
                                break
                    if not secret:
                        secret = json.dumps(data)[:200]
                except Exception:
                    secret = "scrape-empty"
            return secret
        except Exception as e:
            # must return string, not dict
            return f"scrape-error:{e}"

    # -------- PDF QUESTION (generic) --------
    if "pdf" in q_lower and "page 2" in q_lower and "sum" in q_lower:
        pdf_url = extract_first_pdf_link(text, html)
        if not pdf_url:
            return "pdf-error:no-url"
        print(f"[DEBUG] Downloading PDF from: {pdf_url}")
        try:
            pdf_bytes = requests.get(pdf_url, timeout=30).content
            return solve_pdf_sum_value_page2(pdf_bytes)
        except Exception as e:
            return f"pdf-error:{e}"

    # -------- CSV QUESTIONS --------
    if "csv" in q_lower:
        m = re.search(r"(https?://\S+\.csv)", text)
        if m:
            csv_url = m.group(1)
            print(f"[DEBUG] Downloading CSV from: {csv_url}")
            ans = solve_csv_basic(csv_url, q_lower)
            return ans

    # -------- JSON QUESTIONS --------
    if "json" in q_lower:
        m = re.search(r"(https?://\S+\.json)", text)
        if m:
            json_url = m.group(1)
            print(f"[DEBUG] Downloading JSON from: {json_url}")
            ans = solve_json_basic(json_url, q_lower)
            return ans

    # -------- HTML TABLE PARSING --------
    if "table" in q_lower or "html table" in q_lower:
        df = parse_html_table(html)
        if df is not None:
            print(f"[DEBUG] Parsed HTML table with cols: {df.columns.tolist()}")
            if "sum" in q_lower and len(df.columns) > 1:
                col = df.columns[1]
                df[col] = pd.to_numeric(df[col], errors="coerce")
                return float(df[col].sum())
            return df.to_dict(orient="records")

    # -------- Simple linear regression on CSV --------
    if "linear regression" in q_lower:
        m = re.search(r"(https?://\S+\.csv)", text)
        if m:
            csv_url = m.group(1)
            print(f"[DEBUG] Running linear regression on: {csv_url}")
            try:
                df = pd.read_csv(csv_url)
                X = df[[df.columns[0]]].values
                y = df[df.columns[1]].values
                model = LinearRegression().fit(X, y)
                pred = model.predict([[X[-1][0] + 1]])
                return float(pred[0])
            except Exception as e:
                return f"linreg-error:{e}"

    # -------- DEFAULT FALLBACK --------
    print("[WARN] No specific solver matched; returning 'Not-Implemented'")
    return "Not-Implemented"


# ============================================================
# 4. Quiz chain logic – requests only, safe JSON, primitive answers
# ============================================================

def solve_quiz_chain(email: str, secret: str, first_url: str, deadline: datetime):
    """
    Follow the quiz chain: load page, solve, POST to /submit, repeat
    until no next URL or deadline reached.
    All network is done via requests (no Playwright).
    """
    current_url = first_url
    last_result = None

    while current_url and datetime.utcnow() < deadline:
        print("\n==============================")
        print("LOADING:", current_url)
        print("==============================")

        # --- Load quiz page ---
        try:
            resp = requests.get(current_url, timeout=30)
            html = resp.text
            text = html
        except Exception as e:
            print("[ERROR] GET quiz page failed:", e)
            last_result = {
                "status": "page-error",
                "url": current_url,
                "error": str(e),
            }
            break

        print("\n===== CURRENT URL =====")
        print(current_url)
        print("\n===== PAGE TEXT (first 500 chars) =====")
        print(text[:500])
        print("\n===== END PAGE TEXT =====\n")

        # --- Extract question + submit URL ---
        question, submit_url = extract_question_and_submit_url(text, html, current_url)
        print(f"[DEBUG] Submit URL resolved to: {submit_url}")

        if not submit_url:
            last_result = {
                "status": "no-submit-url",
                "url": current_url,
                "error": "Could not resolve submit URL for this quiz page",
            }
            break

        # --- Solve this page ---
        answer = solve_single_quiz(question, text, html, current_url, email)

        # Normalize answer: GA3 expects primitive (str / int / float), not dict/list
        if not isinstance(answer, (str, int, float)):
            answer = str(answer)

        print(f"[DEBUG] Posting answer to: {submit_url}")
        print(f"[DEBUG] Payload answer: {answer!r}")

        # --- POST answer to /submit ---
        try:
            submit_resp = requests.post(
                submit_url,
                json={
                    "email": email,
                    "secret": secret,
                    "url": current_url,
                    "answer": answer,
                },
                timeout=30,
            )
        except Exception as e:
            print("[ERROR] POST to submit URL failed:", e)
            last_result = {
                "status": "post-error",
                "url": current_url,
                "error": f"POST to submit URL failed: {e}",
            }
            break

        # --- Safe JSON parsing: never crash on non-JSON ---
        raw = submit_resp.text or ""
        try:
            data = json.loads(raw)
        except Exception:
            print("[WARN] Response is NOT JSON. Using fallback object.")
            data = {
                "status": "ok",
                "url": None,
                "note": "Non-JSON response from submit endpoint.",
                "raw_response_preview": raw[:300],
            }

        print("\n===== SERVER RESPONSE =====")
        print(json.dumps(data, indent=2))
        print("====================================\n")

        last_result = data
        next_url = data.get("url")
        if not next_url:
            print("[INFO] Quiz chain ended (no next URL).")
            break

        current_url = next_url

    return last_result
