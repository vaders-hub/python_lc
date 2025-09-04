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
    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=600,
        chunk_overlap=200,
        encoding_name='cl100k_base'
    )
    loader = DirectoryLoader(path='./src/static', glob='*.txt', loader_cls=TextLoader)
    
    data = loader.load()
    texts = text_splitter.split_documents(data)
        
    print('split_texts', texts[1].page_content)


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