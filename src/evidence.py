from dataclasses import dataclass, field, asdict

@dataclass
class Evidence:
    """
    Unified evidence object emitted by ALL retrieval sources:
    FAISS (PDFs), PubMed, Logseq, future tools.
    """
    source: str          # "faiss", "pubmed", "logseq"
    id: str              # PMID, chunk hash, page name, etc.
    title: str
    text: str            # Abstract, chunk content, or note body
    relevance: float = 0.0  # Normalized 0.0–1.0 score
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Evidence":
        return Evidence(
            source=d.get("source", "unknown"),
            id=d.get("id", ""),
            title=d.get("title", ""),
            text=d.get("text", d.get("content", "")),
            relevance=d.get("relevance", 0.0),
            meta=d.get("meta", d.get("metadata", {})),
        )
