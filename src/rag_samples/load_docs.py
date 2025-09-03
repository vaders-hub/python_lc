from langchain_community.document_loaders import (DirectoryLoader, 
                                                  TextLoader,
                                                  PyPDFLoader,
                                                  UnstructuredPDFLoader,
                                                  PyMuPDFLoader,
                                                  PyPDFDirectoryLoader,
                                                  OnlinePDFLoader)
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
    data = loader.load()

    print('load_csvs', data[0])


def load_pdfs():
    # pdf_filepath = './src/static/sk.pdf'
    
    # loader = UnstructuredPDFLoader(
    #     pdf_filepath,
    #     strategy="ocr_only",
    #     languages=["kor"],
    #     mode='elements'
    # )
    
    # loader = PyMuPDFLoader(pdf_filepath)
    
    loader = OnlinePDFLoader("https://arxiv.org/pdf/1706.03762.pdf")

    pages = loader.load()
       
    print('pages', pages[0].page_content)