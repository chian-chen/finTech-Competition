# PDF Text Extraction Script

This script is designed to extract text from all PDF files within a specified directory. It loads each PDF file, processes the extracted text, and organizes it into a dictionary for easy retrieval.

---
## Requirements

Python 3.10+

Dependencies:

```
os - For file management.
tqdm - For displaying progress bars.
pdfplumber - For extracting text content from PDF files.
```

---

## Code Overview

The code includes three main functions:

1. load_data():

This function reads all PDF files from a specified folder, extracts their content, and stores each file’s content in a dictionary.
The dictionary keys are file names (without .pdf extension), and the values are the corresponding text content.

2. process_text():

This function is designed to perform any necessary processing on the extracted text, such as cleaning or formatting.
Currently, it returns the text without modifications, but it can be customized for text preprocessing.

3. read_pdf():

This function extracts text from a PDF file, with the option to specify a page range.
If no page range is specified, the function extracts text from all pages.
This function uses pdfplumber to open and read the PDF file.

---
## Usage

1. Load PDF Data:
- Use load_data("path_to_folder") to extract text from all PDFs in the given folder.
- The function returns a dictionary with each PDF’s extracted text, which can be used for further processing or analysis.

2. Process Extracted Text:
- If custom text processing is needed, modify the process_text() function to apply the desired transformations.
