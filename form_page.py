import pathlib

# Option 1: read existing HTML file (recommended if kamu edit form.html separately)
def get_html_from_file(path="./kb-form.html"):
  p = pathlib.Path(path)
  return p.read_text(encoding="utf8")