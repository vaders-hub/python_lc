from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader, UnstructuredPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader


def load_txts():
    loader = DirectoryLoader(path='./src/static', glob='*.txt', loader_cls=TextLoader)

    data = loader.load()
        
    print('load_txts', data[0])
    
    
def load_csvs():
    loader = CSVLoader(
        file_path='./src/static/test.csv',
        encoding='cp949', 
        csv_args={'delimiter': '\n'}
    )
    
    data = loader.load()
    
    print('load_csvs', data[0])


def load_pdfs():
    pdf_filepath = './src/static/sk.pdf'
    loader = UnstructuredPDFLoader(
        pdf_filepath,
        strategy="ocr_only",          # OCR 강제 사용
        languages=["kor"]          # Tesseract 언어 설정
    )

    pages = loader.load()
   
    print('pages', pages)