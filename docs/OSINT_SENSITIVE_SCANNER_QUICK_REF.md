# OSINT Sensitive Scanner - Quick Reference

## ?? Quick Start

```bash
# Install required dependencies
pip install requests beautifulsoup4 tldextract

# Install optional (ENHANCED features)
pip install spacy playwright
python -m spacy download en_core_web_sm
playwright install

# Run scan (basic)
osint_scan_secrets <username>

# Run scan (with JS rendering for LinkedIn, Twitter, etc.)
python resources/python/sherlock_sensitive_scanner.py results.json --use-playwright
```

## ?? Enhanced Features (NOW ENABLED)

### ? spaCy NER (Named Entity Recognition)
- Detects **PERSON** names in content
- Identifies **ORGANIZATION** mentions
- Extracts **LOCATION** (GPE) references
- Finds **DATE** patterns in context
- Links entities to nearby sensitive data

### ? Playwright (JavaScript Rendering)
- Renders modern **single-page apps** (LinkedIn, Twitter, Medium)
- Executes JavaScript before scanning
- Captures **dynamic content** that requests can't see
- Slower but **much more thorough**

## ?? Commands

### Basic Scan
```
osint_scan_secrets <username>
```

### Enhanced Scan (JS Rendering)
```bash
python resources/python/sherlock_sensitive_scanner.py \
    cache/osint/sherlock_username.json \
    --use-playwright
```

### Selective JS Rendering
```bash
python resources/python/sherlock_sensitive_scanner.py \
    cache/osint/sherlock_username.json \
    --js-sites linkedin.com twitter.com reddit.com
```

## ?? What It Finds

### ?? CRITICAL (Act Immediately)
- AWS Keys, Google API Keys
- GitHub PATs, JWT Tokens
- Slack/Stripe Keys
- Private Keys (PEM)
- Passwords in URLs

### ?? HIGH (Review & Remove)
- Phone Numbers
- SSN Patterns
- Credit Card Patterns
- Firebase Endpoints

### ?? MEDIUM (Privacy Concern)
- Email Addresses
- IP Addresses
- High-Entropy Tokens

### ? LOW (Context Dependent)
- Date Patterns
- Generic Patterns

## ?? Output

### Console
```
=== Sensitive Data Exposure Analysis ===
Total Findings: 15
  CRITICAL: 2  HIGH: 5  MEDIUM: 6  LOW: 2
Affected Domains: 4
  • github.com  • pastebin.com
```

### Log File
`cache/osint/sensitive_<username>.jsonl`

## ?? Options

Edit `resources/python/sherlock_sensitive_scanner.py`:

```python
DELAY_BETWEEN = 0.5    # Seconds between requests
FOLLOW_ROBOTS = True   # Respect robots.txt
REQUEST_TIMEOUT = 15   # Request timeout
MAX_CONTEXT = 80       # Context chars
```

## ??? Privacy

? Your accounts only  
? Respects robots.txt  
? Polite scraping  
? Redacted output  
? Local processing  

## ?? If You Find Secrets

1. **Rotate immediately** (API keys, tokens)
2. **Check access logs** for unauthorized use
3. **Enable monitoring**
4. **Remove from public profiles**
5. **Document for audit trail**

## ?? Troubleshooting

| Error | Solution |
|-------|----------|
| "Script not found" | Check `resources/python/sherlock_sensitive_scanner.py` |
| "Missing dependency" | `pip install requests beautifulsoup4 tldextract` |
| "Timeout" | Increase `REQUEST_TIMEOUT` in script |
| "False positives" | Adjust regex patterns in script |

## ?? Related Commands

- `sherlock <user>` - Discover accounts
- `profile_self <user>` - Quick scan
- `osint_report <user>` - Full report
- `osint_status <user>` - Check progress

## ?? Best Practices

1. Run quarterly audits
2. Act on CRITICAL findings immediately
3. Review HIGH findings within 24h
4. Keep logs for compliance
5. Enable 2FA everywhere
6. Monitor HaveIBeenPwned
7. Use separate emails (public/private)
8. Exercise GDPR/CCPA rights

## ?? Full Documentation

- `docs/OSINT_SENSITIVE_SCANNER.md` - Complete guide
- `docs/OSINT_SENSITIVE_SCANNER_SUMMARY.md` - Implementation details
- `docs/OSINT_COMMANDS.md` - All OSINT commands

---

**Remember**: Defensive self-audits only. Use responsibly.
