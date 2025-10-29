#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sherlock_sensitive_scanner.py
Input: sherlock JSON file (site -> { "exists":bool, "url": str, ... })
Output: osint_sensitive_log.jsonl (redacted findings)

Scans discovered URLs for sensitive data:
- Emails, phone numbers, dates of birth
- SSN-like patterns, national IDs
- Credit card patterns (detection only)
- API keys, tokens, secrets (AWS, Google, GitHub, JWT)
- Private keys (PEM blocks)
- High-entropy strings (likely secrets)
- Named entities (people, organizations, locations) with spaCy
- JavaScript-rendered content with Playwright
"""

import re
import json
import time
import math
import argparse
import sys
import os
from urllib.parse import urlparse
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path

try:
    import requests
    from bs4 import BeautifulSoup
    import tldextract
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install requests beautifulsoup4 tldextract")
    sys.exit(1)

# Optional imports
SPACY_AVAILABLE = False
PLAYWRIGHT_AVAILABLE = False

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
    SPACY_AVAILABLE = True
    print("[+] spaCy NER enabled", file=sys.stderr)
except (ImportError, OSError):
    print("[!] spaCy not available - install with: pip install spacy && python -m spacy download en_core_web_sm", file=sys.stderr)

try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_AVAILABLE = True
    print("[+] Playwright JS rendering enabled", file=sys.stderr)
except ImportError:
    print("[!] Playwright not available - install with: pip install playwright && playwright install", file=sys.stderr)


# === Config ===
USER_AGENT = "GRIM-OSINT-Scanner/2.0 (Defensive Self-Audit)"
REQUEST_TIMEOUT = 15
DELAY_BETWEEN = 0.5   # seconds between requests to be polite
MAX_PAGES = 1         # pages per host (keep small)
OUTPUT_FILE = "osint_sensitive_log.jsonl"
VERIFY_SSL = True
FOLLOW_ROBOTS = True   # set False to ignore robots.txt (not recommended)
MAX_CONTEXT = 80

# === Regex detectors (adjust/add as needed) ===
RE_PATTERNS = {
    "email": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", re.I),
    "phone": re.compile(r"(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}"),
    # US SSN pattern (very noisy) - flag for review only
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Date patterns (simple)
    "date": re.compile(r"\b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})\b", re.I),
    # IPv4
    "ipv4": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    # Generic API keys / secrets heuristics (common prefixes)
    "aws_key": re.compile(r"\b(AKIA|ASIA|A3T|AGPA)[A-Z0-9]{16}\b"),
    "google_api_key": re.compile(r"AIza[0-9A-Za-z\-_]{35}"),
    "jwt": re.compile(r"\beyJ[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\.[a-zA-Z0-9-_]+\b"),
    # GitHub/Generic PATs (very heuristic)
    "github_pat": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,255}\b"),
    # PEM private key block
    "pem_priv": re.compile(r"-----BEGIN (?:RSA|EC|DSA|OPENSSH)? PRIVATE KEY-----"),
    # Credit-card-like patterns (16 digits, common separators) - do not assume valid
    "cc_like": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    # Generic high-entropy token fallback: long base64-like strings
    "base64_like": re.compile(r"\b[A-Za-z0-9\-_]{40,}\b"),
    # Slack tokens
    "slack_token": re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"),
    # Stripe keys
    "stripe_key": re.compile(r"(?:sk|pk)_live_[0-9a-zA-Z]{24}"),
    # Firebase
    "firebase": re.compile(r"firebase[a-z0-9_-]{0,30}\.firebaseio\.com"),
    # Generic password in URL
    "password_url": re.compile(r"(?:password|passwd|pwd)=([^&\s]+)", re.I),
}

# === Helper utils ===
def entropy(s: str) -> float:
    """Calculate Shannon entropy per character"""
    if not s:
        return 0.0
    freq = Counter(s)
    probs = [v/len(s) for v in freq.values()]
    return -sum(p * math.log2(p) for p in probs)

def redact_snippet(text: str, match_span, keep=20):
    """Redact matched text but show surrounding context"""
    s, e = match_span
    start = max(0, s - keep)
    end = min(len(text), e + keep)
    snippet = text[start:end]
    # redact center
    left = snippet[:s-start]
    right = snippet[(e-start):]
    return left + "[REDACTED]" + right

def normalize_url(url: str) -> str:
    """Ensure URL has a scheme"""
    if not url:
        return ""
    parsed = urlparse(url)
    if not parsed.scheme:
        return "http://" + url
    return url

def allowed_by_robots(url):
    """Check robots.txt (polite scraping)"""
    if not FOLLOW_ROBOTS:
        return True
    try:
        import urllib.robotparser as robotparser
        p = robotparser.RobotFileParser()
        root = "{uri.scheme}://{uri.netloc}/".format(uri=urlparse(url))
        p.set_url(root + "robots.txt")
        p.read()
        return p.can_fetch(USER_AGENT, url)
    except Exception:
        return True

def extract_text_from_html(html: str):
    """Extract visible text and code blocks from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    # remove scripts/styles
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    # join visible text blocks and code/pre blocks
    texts = []
    for elem in soup.find_all(['pre','code']):
        texts.append(elem.get_text(" ", strip=True))
    texts.append(soup.get_text(" ", strip=True))
    return "\n".join(texts)

def extract_named_entities(text: str):
    """Extract named entities using spaCy (if available)"""
    if not SPACY_AVAILABLE:
        return []
    
    try:
        doc = nlp(text[:100000])  # Limit text length to avoid memory issues
        entities = []
        for ent in doc.ents:
            if ent.label_ in ("PERSON", "ORG", "GPE", "LOC", "DATE"):
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
        return entities
    except Exception as e:
        print(f"[!] spaCy NER error: {e}", file=sys.stderr)
        return []

def fetch_with_playwright(url: str):
    """Fetch URL with JavaScript rendering using Playwright"""
    if not PLAYWRIGHT_AVAILABLE:
        return None
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=USER_AGENT)
            page.goto(url, timeout=REQUEST_TIMEOUT * 1000, wait_until="networkidle")
            content = page.content()
            browser.close()
            return content
    except Exception as e:
        print(f"[!] Playwright error for {url}: {e}", file=sys.stderr)
        return None

def severity_for_tag(tag, match_text, ent):
    """Assign severity score (0-10)"""
    if tag in ("pem_priv", "aws_key", "google_api_key", "jwt", "github_pat", "slack_token", "stripe_key"):
        return 10  # CRITICAL - immediate action required
    if tag in ("ssn", "cc_like", "password_url"):
        return 9   # HIGH - likely PII or credentials
    if tag in ("email", "phone"):
        return 7 if tag == "phone" else 6  # MEDIUM - contact info
    if tag == "date":
        return 4   # LOW - might be DOB in context
    if tag == "firebase":
        return 8   # HIGH - exposed service
    if ent and ent > 4.5:
        return 9   # HIGH - very high entropy suggests secret
    if ent and ent > 4.0:
        return 7   # MEDIUM-HIGH
    return 5       # MEDIUM - generic finding

# === Core scanning ===
def scan_url(url, session, use_playwright=False):
    """Scan a single URL for sensitive data"""
    url = normalize_url(url)
    findings = []
    
    if not allowed_by_robots(url):
        return findings, "robots_blocked"
    
    try:
        # Try Playwright for JS-heavy sites if available
        html = None
        if use_playwright and PLAYWRIGHT_AVAILABLE:
            html = fetch_with_playwright(url)
        
        # Fallback to requests if Playwright failed or not used
        if html is None:
            r = session.get(url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, verify=VERIFY_SSL)
            r.raise_for_status()
            html = r.text
        
        text = extract_text_from_html(html)
    except Exception as e:
        return findings, f"error:{str(e)[:50]}"

    # Extract named entities if spaCy is available
    entities = extract_named_entities(text) if SPACY_AVAILABLE else []
    
    # run regexes
    for tag, rx in RE_PATTERNS.items():
        for m in rx.finditer(text):
            raw = m.group(0)
            ent = entropy(raw)
            sev = severity_for_tag(tag, raw, ent)
            
            # Redact high-risk matches completely
            should_redact = tag in ("pem_priv", "aws_key", "google_api_key", "jwt", 
                                   "github_pat", "slack_token", "stripe_key", "password_url")
            
            red = redact_snippet(text, m.span(), keep=MAX_CONTEXT)
            
            # Check if this match is near a named entity (adds context)
            nearby_entity = None
            if entities:
                for entity in entities:
                    if abs(entity["start"] - m.start()) < 200:  # Within 200 chars
                        nearby_entity = f"{entity['label']}:{entity['text']}"
                        break
            
            finding = {
                "tag": tag,
                "match": "[REDACTED]" if should_redact else raw,
                "context": red,
                "entropy": round(ent, 2),
                "severity": sev,
                "location": url,
                "match_length": len(raw)
            }
            
            if nearby_entity:
                finding["nearby_entity"] = nearby_entity
            
            findings.append(finding)

    # fallback: long base64-like tokens flagged by entropy
    for m in RE_PATTERNS["base64_like"].finditer(text):
        candidate = m.group(0)
        ent = entropy(candidate)
        if ent > 4.0 and len(candidate) > 40:
            # Skip if already matched by more specific regex
            already_found = any(f["match"] == candidate or 
                              (f["match"] == "[REDACTED]" and 
                               abs(f["match_length"] - len(candidate)) < 5)
                              for f in findings)
            if not already_found:
                findings.append({
                    "tag": "high_entropy_token",
                    "match": "[REDACTED]",
                    "context": redact_snippet(text, m.span(), keep=MAX_CONTEXT),
                    "entropy": round(ent, 2),
                    "severity": severity_for_tag("high_entropy_token", candidate, ent),
                    "location": url,
                    "match_length": len(candidate)
                })
    
    # Add named entity findings (PERSON entities near dates might be DOB context)
    if SPACY_AVAILABLE and entities:
        person_entities = [e for e in entities if e["label"] == "PERSON"]
        if person_entities:
            for person in person_entities[:5]:  # Limit to first 5 names
                findings.append({
                    "tag": "person_name",
                    "match": person["text"],
                    "context": redact_snippet(text, (person["start"], person["end"]), keep=MAX_CONTEXT),
                    "entropy": 0,
                    "severity": 6,  # Medium severity - might be your name
                    "location": url,
                    "match_length": len(person["text"]),
                    "entity_type": "PERSON"
                })

    return findings, "ok"

def load_sherlock_json(path):
    """Load Sherlock output and extract URLs"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    urls = []
    # Handle various Sherlock output formats
    if isinstance(data, dict):
        for site, info in data.items():
            if isinstance(info, dict):
                # Standard format: { "status": "Claimed", "url": "..." }
                if info.get("url") and info.get("status") == "Claimed":
                    urls.append(info["url"])
                # Alternative: { "exists": true, "url": "..." }
                elif info.get("exists") and info.get("url"):
                    urls.append(info["url"])
                # URL-only format
                elif info.get("url_main"):
                    urls.append(info["url_main"])
            elif isinstance(info, str):
                urls.append(info)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and item.get("url"):
                urls.append(item["url"])
    
    return urls

def generate_summary_report(log_file):
    """Generate a summary report from the log file"""
    if not os.path.exists(log_file):
        return
    
    findings_by_severity = defaultdict(list)
    findings_by_tag = defaultdict(int)
    domains_with_findings = set()
    
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                for finding in entry.get("findings", []):
                    sev = finding["severity"]
                    tag = finding["tag"]
                    findings_by_severity[sev].append(finding)
                    findings_by_tag[tag] += 1
                    if finding.get("location"):
                        domain = tldextract.extract(finding["location"]).registered_domain
                        domains_with_findings.add(domain)
            except json.JSONDecodeError:
                continue
    
    # Print summary
    print("\n" + "="*60)
    print("OSINT SENSITIVE DATA SCAN SUMMARY")
    print("="*60)
    
    print(f"\nTotal Findings: {sum(findings_by_tag.values())}")
    print(f"Affected Domains: {len(domains_with_findings)}")
    
    print("\n--- By Severity ---")
    for sev in sorted(findings_by_severity.keys(), reverse=True):
        count = len(findings_by_severity[sev])
        level = "CRITICAL" if sev >= 9 else "HIGH" if sev >= 7 else "MEDIUM" if sev >= 5 else "LOW"
        print(f"  [{level:8s}] Severity {sev}: {count} findings")
    
    print("\n--- By Type ---")
    for tag, count in sorted(findings_by_tag.items(), key=lambda x: x[1], reverse=True):
        print(f"  {tag:20s}: {count}")
    
    print("\n--- Affected Domains ---")
    for domain in sorted(domains_with_findings):
        print(f"  � {domain}")
    
    print("\n" + "="*60)
    print(f"Full log saved to: {log_file}")
    print("="*60 + "\n")

def main():
    p = argparse.ArgumentParser(description="Scan Sherlock results for sensitive data exposure")
    p.add_argument("sherlock_json", help="Sherlock JSON output file")
    p.add_argument("--out", default=OUTPUT_FILE, help="Output JSONL file")
    p.add_argument("--no-delay", action="store_true", help="Skip delays between requests")
    p.add_argument("--max", type=int, default=MAX_PAGES, help="Max pages per domain")
    p.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT, help="Request timeout in seconds")
    p.add_argument("--no-robots", action="store_true", help="Ignore robots.txt")
    p.add_argument("--use-playwright", action="store_true", help="Use Playwright for JS rendering (slower but more thorough)")
    p.add_argument("--js-sites", nargs="*", help="Specific domains to use Playwright for (e.g., linkedin.com twitter.com)")
    args = p.parse_args()

    # Override globals with args
    REQUEST_TIMEOUT = args.timeout if args.timeout != 15 else REQUEST_TIMEOUT`n    FOLLOW_ROBOTS = not args.no_robots

    # Load URLs from Sherlock output
    try:
        urls = load_sherlock_json(args.sherlock_json)
    except FileNotFoundError:
        print(f"Error: Sherlock JSON file not found: {args.sherlock_json}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {args.sherlock_json}: {e}")
        return 1

    if not urls:
        print("Warning: No URLs found in Sherlock output")
        return 0

    print(f"[+] Loaded {len(urls)} URLs from Sherlock output")
    print(f"[+] Output file: {args.out}")
    print(f"[+] User-Agent: {USER_AGENT}")
    if SPACY_AVAILABLE:
        print(f"[+] spaCy NER: ENABLED (will detect names, dates, locations)")
    if PLAYWRIGHT_AVAILABLE:
        print(f"[+] Playwright: ENABLED (will render JavaScript)")
    print()

    # Build list of JS-heavy sites
    js_sites = set(args.js_sites or [])
    if args.use_playwright and PLAYWRIGHT_AVAILABLE:
        # Common JS-heavy sites that benefit from Playwright
        js_sites.update([
            "linkedin.com", "twitter.com", "facebook.com", "instagram.com",
            "reddit.com", "medium.com", "dev.to", "stackoverflow.com"
        ])

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    
    total_findings = 0
    with open(args.out, "a", encoding="utf-8") as outfh:
        for i, url in enumerate(urls, 1):
            parsed = urlparse(url)
            domain = tldextract.extract(url).registered_domain
            
            # Check if this domain should use Playwright
            use_pw = domain in js_sites if js_sites else args.use_playwright
            
            if use_pw and PLAYWRIGHT_AVAILABLE:
                print(f"[{i}/{len(urls)}] Scanning (JS): {domain}")
            else:
                print(f"[{i}/{len(urls)}] Scanning: {domain}")
            
            try:
                findings, status = scan_url(url, session, use_playwright=use_pw)
            except Exception as e:
                findings, status = [], f"exception:{str(e)[:50]}"
            
            out = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "source_url": url,
                "domain": domain,
                "status": status,
                "rendered_with_js": use_pw and PLAYWRIGHT_AVAILABLE,
                "used_ner": SPACY_AVAILABLE,
                "findings": findings
            }
            outfh.write(json.dumps(out, ensure_ascii=False) + "\n")
            
            if findings:
                total_findings += len(findings)
                max_sev = max(f['severity'] for f in findings)
                print(f"  ? {len(findings)} findings (max severity: {max_sev})")
            else:
                print(f"  ? Clean")
            
            if not args.no_delay and i < len(urls):
                time.sleep(DELAY_BETWEEN)

    print(f"\n[+] Scan complete: {total_findings} total findings")
    
    # Generate summary
    generate_summary_report(args.out)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())


