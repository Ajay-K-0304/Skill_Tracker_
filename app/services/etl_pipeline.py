import hashlib
from bs4 import BeautifulSoup

def clean_html(raw_html: str) -> str:
    """Strips HTML tags and returns clean raw text strings."""
    if not raw_html:
        return ""
    
    # Parse HTML and extract text with spaces replacing the tags
    soup = BeautifulSoup(raw_html, "html.parser")
    clean_text = soup.get_text(separator=" ", strip=True)
    
    return clean_text

def generate_job_hash(title: str, company: str, location: str) -> str:
    """
    Generates a unique hash to check if a job already exists in the database.
    Normalizes the string to lower case to prevent duplicate case-mismatches.
    """
    # Fallback to empty string if any value is None
    title = title or ""
    company = company or ""
    location = location or ""
    
    unique_string = f"{title.strip()}|{company.strip()}|{location.strip()}".lower()
    return hashlib.sha256(unique_string.encode('utf-8')).hexdigest()