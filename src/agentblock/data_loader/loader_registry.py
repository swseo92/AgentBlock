from agentblock.data_loader.simple_loader import simple_file_loader, simple_api_loader
from agentblock.data_loader.document_loader import pdf_loader


LOADER_IMPL_MAP = {
    "simple_file_loader": simple_file_loader,
    "simple_api_loader": simple_api_loader,
    "pdf_loader": pdf_loader,
}
