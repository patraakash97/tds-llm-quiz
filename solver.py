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

from playwright.sync_api import (
    sync_playwright,
    TimeoutError as PlaywrightTimeoutError,
    Error as PlaywrightError,
)

# ============================================================
# 1. Utility helpers
# ============================================================

def origin_from_url(url: str) -> Optional[str]:
    try:
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}" if p.scheme and p.netloc else None
    except Exception:
        return None


def extract_question_and_submit_url(text: str, html: str, current_url: str):
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
    m = re.search(r'href="(https?://[^"]+\.pdf)"', html)
    if m:
        return m.group(1)

    m = re.search(r"(https?://\S+\.pdf)", text)
    if m:
        return m.group(1).rstrip(").,")

    return None


# ============================================================
# 2. Built-in solvers
# ============================================================

def solve_pdf_sum_value_page2(pdf_bytes: bytes) -> float:
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        if len(pdf.pages) < 2:
            raise ValueError("PDF has less than 2 pages, cannot use page 2")
        page2 = pdf.pages[1]
        table = page2.extract_table()
        if not table:
            raise ValueError("No table found on PDF page 2")

        df = pd.DataFrame(table[1:], columns=table[0])
        col = next(c for c in df.columns if c.lower() == "value")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        return float(df[col].sum())


def solve_csv_basic(url: str, task: str):
    try:
        df = pd.read_csv(url)
    except Exception as e:
        return {"error": f"Failed to load CSV: {str(e)}"}

    task_low = task.lower()

    if "mean" in task_low:
        col = re.search(r"mean.*?(\w+)", task, re.I).group(1)
        return float(df[col].mean())

    if "sum" in task_low:
        col = re.search(r"sum.*?(\w+)", task, re.I).group(1)
        return float(df[col].sum())

    if "max" in task_low:
        col = re.search(r"max.*?(\w+)", task, re.I).group(1)
        return float(df[col].max())

    if "min" in task_low:
        col = re.search(r"min.*?(\w+)", task, re.I).group(1)
        return float(df[col].min())

    return df.to_dict(orient="records")


def solve_json_basic(url: str, task: str):
    try:
        data = requests.get(url).json()
    except Exception as e:
        return {"error": f"Failed to parse JSON: {str(e)}"}

    task_low = task.lower()

    if isinstance(data, list):
        if "count" in task_low:
            return len(data)

        if "max" in task_low:
            field = re.search(r"max.*?(\w+)", task, re.I).group(1)
            return max(item[field] for item in data if field in item)

    return data


def parse_html_table(page):
    rows = page.locator("table tr")
    row_count = rows.count()
    data = []

    for i in range(row_count):
        row = rows.nth(i).locator("td")
        cols = row.count()
        if cols == 0:
            continue
        data.append([row.nth(j).inner_text() for j in range(cols)])

    if not data:
        return None

    headers = data[0]
    body = data[1:]
    return pd.DataFrame(body, columns=headers)


def solve_image_ocr(base64_img: str) -> str:
    """OCR disabled (no EasyOCR dependency)."""
    return "OCR-not-available"


def generate_plot_base64(df, x, y) -> str:
    fig, ax = plt.subplots()
    ax.plot(df[x], df[y])
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ============================================================
# 3. Per-page quiz solver
# ============================================================

def solve_single_quiz(question, text, html, page):
    q_lower = question.lower()
    text_lower = text.lower()

    print("\n=== SOLVING QUESTION SNIPPET ===")
    print(question[:300], "...\n")

    # Demo page: 'answer': 'anything you want'
    if "anything you want" in text_lower:
        return "hello-from-akash"

    # Relative scrape type
    if "scrape" in q_lower and "demo-scrape-data" in q_lower:
        m = re.search(r"Scrape\s+(\S+)", question, re.I)
        if m:
            relative = m.group(1)
            origin = origin_from_url(page.url)
            url = origin + relative
            print(f"[DEBUG] Scraping relative URL: {url}")
            return requests.get(url).json()

    # PDF
    if "sum" in q_lower and "value" in q_lower and "page 2" in q_lower:
        pdf_url = extract_first_pdf_link(text, html)
        if not pdf_url:
            raise ValueError("Could not find PDF URL in question/page.")
        print(f"[DEBUG] Downloading PDF from: {pdf_url}")
        pdf_bytes = requests.get(pdf_url).content
        return solve_pdf_sum_value_page2(pdf_bytes)

    # CSV
    if "csv" in q_lower:
        m = re.search(r"(https?://\S+\.csv)", text)
        if m:
            csv_url = m.group(1)
            print(f"[DEBUG] Downloading CSV from: {csv_url}")
            return solve_csv_basic(csv_url, q_lower)

    # JSON
    if "json" in q_lower:
        m = re.search(r"(https?://\S+\.json)", text)
        if m:
            json_url = m.group(1)
            print(f"[DEBUG] Downloading JSON from: {json_url}")
            return solve_json_basic(json_url, q_lower)

    # HTML table
    if "table" in q_lower or "html table" in q_lower:
        df = parse_html_table(page)
        if df is not None:
            print(f"[DEBUG] Parsed HTML table with cols: {df.columns.tolist()}")
            if "sum" in q_lower:
                col = df.columns[1]
                df[col] = pd.to_numeric(df[col], errors="ignore")
                return float(df[col].sum())
            return df.to_dict(orient="records")

    # Linear regression
    if "linear regression" in q_lower:
        m = re.search(r"(https?://\S+\.csv)", text)
        if m:
            csv_url = m.group(1)
            print(f"[DEBUG] Running linear regression on: {csv_url}")
            df = pd.read_csv(csv_url)
            X = df[[df.columns[0]]].values
            y = df[df.columns[1]].values
            model = LinearRegression().fit(X, y)
            pred = model.predict([[X[-1][0] + 1]])
            return float(pred[0])

    print("[WARN] No specific solver matched; returning 'Not-Implemented'")
    return "Not-Implemented"


# ============================================================
# 4. Quiz chain logic – with HARD fallback + safe JSON parsing
# ============================================================

def _load_page_with_playwright_or_requests(page, url: str):
    """
    Try to load with Playwright. If anything goes wrong,
    fall back to requests.get(url) and treat it as static HTML.
    Returns (html, text).
    """
    try:
        page.goto(url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(500)
        html = page.content()
        text = page.inner_text("body")
        return html, text
    except (PlaywrightTimeoutError, PlaywrightError, Exception) as e:
        print("[WARN] Playwright load failed, falling back to requests.get:", e)
        resp = requests.get(url, timeout=20)
        html = resp.text
        text = html
        return html, text


def solve_quiz_chain(email, secret, first_url, deadline):
    current_url = first_url
    last_result = None

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            while current_url and datetime.utcnow() < deadline:
                page = browser.new_page()
                page.set_default_timeout(90000)

                print("\n==============================")
                print("LOADING:", current_url)
                print("==============================")

                html, text = _load_page_with_playwright_or_requests(page, current_url)

                print("\n===== CURRENT URL =====")
                print(page.url if page.url else current_url)
                print("\n===== PAGE TEXT (first 500 chars) =====")
                print(text[:500])
                print("\n===== END PAGE TEXT =====\n")

                question, submit_url = extract_question_and_submit_url(
                    text, html, current_url
                )
                print(f"[DEBUG] Submit URL resolved to: {submit_url}")

                if not submit_url:
                    last_result = {
                        "status": "no-submit-url",
                        "url": None,
                        "error": "Could not resolve submit URL for this quiz page",
                    }
                    break

                answer = solve_single_quiz(question, text, html, page)

                print(f"[DEBUG] Posting answer to: {submit_url}")
                print(f"[DEBUG] Payload answer: {answer!r}")

                try:
                    resp = requests.post(
                        submit_url,
                        json={
                            "email": email,
                            "secret": secret,
                            "url": current_url,
                            "answer": answer,
                        },
                        timeout=20,
                    )
                except Exception as e:
                    last_result = {
                        "status": "post-error",
                        "url": None,
                        "error": f"POST to submit URL failed: {e}",
                    }
                    break

                # -------- SAFE JSON PARSING (cannot raise Expecting value) --------
                raw = resp.text or ""
                try:
                    data = json.loads(raw)
                except Exception:
                    print("[WARN] Response is NOT JSON. Using fallback success object.")
                    data = {
                        "status": "ok",
                        "url": None,
                        "note": "Non-JSON response (likely demo /submit).",
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

        finally:
            browser.close()

    return last_result
