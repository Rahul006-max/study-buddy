import PyPDF2
from io import BytesIO

def extract_text_from_pdf(uploaded_file):
    """
    Extracts text from a PDF file uploaded via Streamlit.
    
    Args:
        uploaded_file: The file object from st.file_uploader.
        
    Returns:
        str: The extracted text content.
    """
    try:
        # Use BytesIO to treat the uploaded file as a file-like object
        pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
        text = ""
        
        # Iterate through each page and extract text
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
                
        return text.strip()
    except Exception as e:
        raise ValueError(f"Error reading PDF: {str(e)}")