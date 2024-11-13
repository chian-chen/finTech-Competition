import os  # For interacting with the file system
from tqdm import tqdm  # For displaying progress bars
import pdfplumber  # Library for extracting text from PDF files

# Function to load PDF data from a directory and store it in a dictionary
def load_data(source_path):
    """
    Loads and processes all PDF files from a specified directory.
    
    Parameters:
        source_path (str): Path to the directory containing PDF files.

    Returns:
        dict: A dictionary where the keys are file names (without .pdf extension)
              and values are the extracted text content of the PDFs.
    """
    # Get the list of files in the specified directory
    masked_file_ls = os.listdir(source_path)
    
    # Read and extract text from each PDF, using the file name (without extension) as the key
    corpus_dict = {
        int(file.replace(".pdf", "")): read_pdf(os.path.join(source_path, file))
        for file in tqdm(masked_file_ls)
    }
    
    return corpus_dict

# Function to process the extracted text (can be customized for text preprocessing)
def process_text(text):
    """
    Processes the extracted text. This function can be customized to clean or transform
    the text as needed.

    Parameters:
        text (str): The extracted text.

    Returns:
        str: The processed text.
    """
    return text  # Currently returns the text as-is

# Function to read a PDF and extract text from specified pages or all pages
def read_pdf(pdf_loc, page_infos: list = None):
    """
    Reads and extracts text from a PDF file. Can extract text from specific page ranges.
    
    Parameters:
        pdf_loc (str): Path to the PDF file.
        page_infos (list, optional): A list containing the start and end pages to extract.
                                     If None, extracts text from all pages.

    Returns:
        str: The extracted text from the PDF.
    """
    pdf = pdfplumber.open(pdf_loc)  # Open the PDF file

    # Select pages based on the specified range; if None, selects all pages
    pages = pdf.pages[page_infos[0] : page_infos[1]] if page_infos else pdf.pages
    pdf_text = ""

    # Iterate over each page and extract text
    for _, page in enumerate(pages):
        text = page.extract_text()  # Extract text from the page
        if text:
            pdf_text += process_text(text)  # Process and append text to pdf_text

    pdf.close()  # Close the PDF file after extraction

    return pdf_text  # Return the combined text from all pages
