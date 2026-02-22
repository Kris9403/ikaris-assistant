import sys
from unittest.mock import MagicMock

# List of heavy dependencies to mock if missing
# Perform mocking in a hook to ensure it runs early
def pytest_configure(config):
    print("DEBUG: Running pytest_configure in conftest.py")
    # List of heavy dependencies to mock if missing
    MOCK_MODULES = [
        'langchain_community',
        'langchain_community.document_loaders',
        'langchain_text_splitters',
        'langchain_huggingface',
        'langchain_community.vectorstores',
        'langchain_openai',
        'langchain_core',
        'langchain_core.messages',
        'langgraph',
        'langgraph.graph',
        'langgraph.graph.message',
        'langgraph.checkpoint',
        'langgraph.checkpoint.sqlite',
        'faiss', 
        'arxiv',
        'requests',
        'psutil',
        'dotenv',
        'rich',
        'markdown_it',
    ]

    # Attempt to import. If missing, mock it.
    # Start with top-level packages to avoid parent package errors
    sorted_modules = sorted(MOCK_MODULES, key=lambda x: x.count('.'))

    for mod_name in sorted_modules:
        try:
            __import__(mod_name)
        except ImportError:
            sys.modules[mod_name] = MagicMock()
            print(f"DEBUG: Mocked {mod_name}")
