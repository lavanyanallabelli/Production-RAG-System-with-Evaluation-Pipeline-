from pypdf import PdfReader

#file path in string and expected output is string
def extract_text_from_pdf(file_path: str) -> str: 
    #PdfReader is a class in pypdf library that reads a PDF file and extracts the text from it
    reader = PdfReader(file_path)
    #empty list to store the text from each page
    pages = []
    #iterate through each page in the PDF file
    for page in reader.pages:
        text = page.extract_text()
        #if text is not empty and not whitespace, add it to the list
        if text and text.strip():
            pages.append(text.strip())

    if not pages:
        raise ValueError("Could not extract any text from this PDF")
    #join the text from each page with a newline and a newline
    full_text = "\n\n".join(pages)
    #return the full text
    return full_text

    #test in directory, file path is the file name in the same directory as the script
    #__name__ is a special variable in Python that is set to the name of the module.
    #__main__ is the name of the main module.
if __name__ == "__main__":
    text = extract_text_from_pdf("OnePiece.pdf")

    print(f"Total characters extracted: {len(text)}")
    print(f"First 500 characters:\n")
    print(text[:500])