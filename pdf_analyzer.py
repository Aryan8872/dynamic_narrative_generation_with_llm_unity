import PyPDF2
import os

def analyze_pdf(pdf_path):
    """Analyze a PDF file and extract its content."""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            print(f"PDF Analysis Results:")
            print(f"File: {os.path.basename(pdf_path)}")
            print(f"Number of pages: {len(pdf_reader.pages)}")
            print(f"File size: {os.path.getsize(pdf_path)} bytes")
            print("-" * 50)
            
            # Extract text from all pages
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages, 1):
                text = page.extract_text()
                full_text += f"\n--- Page {page_num} ---\n{text}\n"
            
            return full_text
            
    except Exception as e:
        print(f"Error analyzing PDF: {e}")
        return None

if __name__ == "__main__":
    pdf_file = "Refactored Adaptive Story & NPC System Design.pdf"
    content = analyze_pdf(pdf_file)
    
    if content:
        print("PDF Content:")
        print(content)
        
        # Save content to a text file for easier reading
        with open("pdf_content.txt", "w", encoding="utf-8") as f:
            f.write(content)
        print("\nContent has been saved to 'pdf_content.txt'") 