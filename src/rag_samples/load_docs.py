from langchain_community.document_loaders import (DirectoryLoader, 
                                                  TextLoader,
                                                  PyPDFLoader,
                                                  UnstructuredPDFLoader,
                                                  PyMuPDFLoader,
                                                  PyPDFDirectoryLoader,
                                                  OnlinePDFLoader)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import CharacterTextSplitter


def load_txts():
    text_splitter = CharacterTextSplitter(
        separator='',
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    loader = DirectoryLoader(path='./src/static', glob='*.txt', loader_cls=TextLoader)
    
    data = loader.load()
    texts = text_splitter.split_text(data[0].page_content)
        
    print('split_texts', texts[0])


def load_csvs():
    loader = CSVLoader(
        file_path='./src/static/test.csv',
        encoding='cp949', 
        csv_args={'delimiter': '\n'}
    )
    
    data = loader.load()
    
    print('load_csvs', data)


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