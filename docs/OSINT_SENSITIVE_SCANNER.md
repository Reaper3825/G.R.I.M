# OSINT Sensitive Data Scanner

## Overview

The sensitive data scanner is an advanced feature that scans your discovered online profiles for exposed sensitive information. It helps identify:

- **Personal Information**: Emails, phone numbers, dates of birth
- **Identity Documents**: SSN patterns, passport/license numbers
- **Financial Data**: Credit card patterns (detection only, not validation)
- **Secrets & Credentials**: API keys, tokens, passwords, private keys
- **High-Entropy Data**: Base64-encoded secrets, JWT tokens

## ?? Installation

### Required Python Packages

```bash
pip install requests beautifulsoup4 tldextract
```

### Optional (Enhanced Features)

```bash
# For named entity recognition (names, locations, dates)
pip install spacy
python -m spacy download en_core_web_sm

# For JavaScript-heavy sites (LinkedIn, Twitter, etc.)
pip install playwright
playwright install
```

## ?? Usage

### Basic Command

```
osint_scan_secrets <username>
```

### Enhanced Scan (with JS rendering)

```bash
# Scan with Playwright for JavaScript-heavy sites
python resources/python/sherlock_sensitive_scanner.py \
    cache/osint/sherlock_<username>.json \
    --use-playwright

# Scan specific sites with JS rendering
python resources/python/sherlock_sensitive_scanner.py \
    cache/osint/sherlock_<username>.json \
    --js-sites linkedin.com twitter.com reddit.com
```

### Natural Language

```
"scan for secrets for john_doe"
"check for sensitive data exposed for alice"
"find leaked information for bob_smith"
"detect exposed secrets"
```

### Workflow

1. **Run OSINT Scan First** (if not cached):
   ```
   sherlock john_doe
   ```

2. **Basic Scan for Sensitive Data**:
   ```
   osint_scan_secrets john_doe
   ```

3. **Enhanced Scan (JS + NER)**:
   ```bash
   python resources/python/sherlock_sensitive_scanner.py \
       cache/osint/sherlock_john_doe.json \
       --use-playwright \
       --out cache/osint/sensitive_john_doe_enhanced.jsonl
   ```

4. **Review Results**:
   - Output shows summary with severity levels
   - Full log saved to `cache/osint/sensitive_<username>.jsonl`

## ?? What It Detects

### Critical (Severity 9-10)
- ?? **Private Keys**: RSA, EC, DSA, OpenSSH private keys
- ?? **AWS Keys**: AKIA, ASIA, A3T, AGPA patterns
- ?? **Google API Keys**: AIza... patterns
- ?? **GitHub Personal Access Tokens**: ghp_, gho_, ghu_ patterns
- ?? **JWT Tokens**: eyJ... patterns
- ?? **Slack Tokens**: xoxb-, xoxa-, xoxp- patterns
- ?? **Stripe Keys**: sk_live_, pk_live_ patterns
- ?? **Passwords in URLs**: password=... patterns

### High (Severity 7-8)
- ?? **Phone Numbers**: International and local formats
- ?? **SSN Patterns**: ###-##-#### format
- ?? **Credit Card Patterns**: 13-19 digit sequences
- ?? **Firebase Endpoints**: *.firebaseio.com
- ?? **High-Entropy Strings**: Base64-like tokens with >4.5 bits/char

### Medium (Severity 5-6)
- ?? **Email Addresses**: username@domain.com
- ?? **High-Entropy Tokens**: >4.0 bits/char
- ?? **IPv4 Addresses**: Public IP exposure
- ?? **Person Names** (spaCy): Names detected in content (may be yours)

### Low (Severity 0-4)
- ? **Date Patterns**: Possible DOB in context
- ? **Generic Patterns**: May require manual review

## ?? Enhanced Detection (with Optional Dependencies)

### spaCy Named Entity Recognition
When spaCy is installed, the scanner also detects:

- **PERSON**: Full names in content (yours or others mentioned)
- **ORG**: Organizations, companies mentioned
- **GPE**: Geopolitical entities (cities, countries)
- **DATE**: Dates in natural language
- **Entity Context**: Links entities to nearby sensitive data

**Example**: If it finds "John Smith" near a phone number, it flags both with context.

### Playwright JavaScript Rendering
When Playwright is installed, the scanner can:

- **Render JavaScript**: See content that requests.get() can't
- **Handle SPAs**: Modern single-page applications (React, Vue, Angular)
- **Dynamic Content**: Content loaded via AJAX/fetch
- **Auth Walls**: Content behind "log in to see more" (if publicly viewable)

**Best for**: LinkedIn, Twitter, Facebook, Medium, Dev.to, Reddit
