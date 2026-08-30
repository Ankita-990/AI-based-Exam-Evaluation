# 📝 Answer Sheet Mapper

An AI-based examiner assistant that reads a question paper and a handwritten answer sheet, then automatically matches each answer to its question and shows exactly where it sits on the sheet.
EasyOCR extracts text from both the printed question paper and the handwritten answer sheet. Matching happens in two passes: answers are first paired directly by label — Q1 ↔ Ans1/Answer1, 11(a) ↔ 11(a), and so on. When an answer has no clear label, spaCy steps in as a fallback, comparing the meaning of the answer against each remaining question using cosine similarity between their word vectors, and only accepting a match above a conservative confidence threshold — anything lower is flagged as unmatched instead of guessed. Streamlit powers the interface and lets the whole thing run in the browser, deployed publicly on Streamlit Community Cloud with nothing for the user to install.

## Try it locally

pip install -r requirements.txt

streamlit run app.py


### You'll need Poppler installed for PDF support 
#### Mac
brew install poppler 

#### Linux
apt-get install poppler-utils

#### Windows build
Download the zip from the link below and extract it contains to the specific folder👇
https://github.com/oschwartz10612/poppler-windows/releases
