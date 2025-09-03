# from lc_samples import lang_default
# from lc_samples import lang_prompt
# from lc_samples import lang_partial
# from lc_samples import lang_parser
# from lc_samples import lang_memory
# from rag_samples import load_data
from rag_samples import load_docs


def start():
    load_docs.load_pdfs()
