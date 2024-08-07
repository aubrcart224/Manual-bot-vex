import PyPDF2 


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfFileReader(file)
        text = ''
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extract_text()
        return text
    

manual_text = extract_text_from_pdf('manual.pdf')
with open('manual.txt', 'w') as text_file:
    text_file.write(manual_text)

    