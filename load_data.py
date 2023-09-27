from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from config import CFG

loader = DirectoryLoader(
    CFG.PDFs_path,
    glob="./*.pdf",
    loader_cls=PyPDFLoader,
    show_progress=True,
    use_multithreading=True
)

documents = loader.load()