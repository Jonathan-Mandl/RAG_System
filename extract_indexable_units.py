import glob
import json
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter
from bidi.algorithm import get_display
import re

class CustomConverter(MarkdownConverter):
    def convert_a(self, el, text, convert_as_inline):
        return text

    def convert_table(self, el, text, convert_as_inline):
        return "[טבלה]"

def process_html_to_indexable_unit(fname):
    """Processes a single HTML file into an indexable unit."""
    doc = BeautifulSoup(open(fname, encoding="utf-8").read(), "html.parser")
    doc_id = fname.split("/")[-1].replace(".html", "")
    
    # Extract title
    title = doc.title.string if doc.title else "No Title"
    
    # For debugging, use get_display
    # print("Title (for debugging):")
    # print(get_display(title))
    
    # Extract main content and convert to Markdown
    main = doc.main
    if not main:
        return None  # Skip documents with no main content

    as_md = CustomConverter(heading_style="ATX", bullets="*").convert_soup(main)
    as_md = re.sub(r"\n\n+", "\n\n", as_md)

    # Split Markdown into sections
    sections = []
    for section in as_md.split("\n#"):
        if not section.strip():
            continue
        section = "#" + section  # Add back the removed '#'
        sec_title, sec_body = section.split("\n", 1)
        
        # Debugging: Display in terminal
        # print(f"SEC--{get_display(sec_title)}--------------------")
        # print(get_display(sec_body))
        
        # Save sections without get_display
        sections.append({
            "section_title": sec_title.strip(),
            "section_body": sec_body.strip()
        })

    # Return indexable unit
    return {
        "doc_id": doc_id,
        "title": title.strip(),
        "sections": sections
}

# Process all HTML files and save to JSONL
indexable_units = []
for i,fname in enumerate(glob.glob("pages/*.html")):
    if i%100 == 0:
        print(i)
    unit = process_html_to_indexable_unit(fname)
    if unit:
        indexable_units.append(unit)

print("Done extraction!")

# Save to JSONL
with open("indexed_content.jsonl", "w", encoding="utf-8") as f:
    for unit in indexable_units:
        f.write(json.dumps(unit, ensure_ascii=False) + "\n")
