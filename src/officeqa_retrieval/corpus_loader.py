import logging
import re
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

TOKEN_PATTERN = re.compile(r"[a-zA-Z]{3,}")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")


@dataclass(frozen=True)
class CorpusDocument:
    path: Path
    title: str
    content: str
    tokens: set[str]
    years: set[str]


def _tokenize(text: str) -> set[str]:
    return {match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)}


def load_corpus(corpus_dir: Path | None) -> list[CorpusDocument]:
    if corpus_dir is None:
        return []
    if not corpus_dir.exists():
        logger.warning("Corpus directory does not exist: %s", corpus_dir)
        return []

    documents: list[CorpusDocument] = []
    for path in sorted(corpus_dir.rglob("*.txt")):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            logger.warning("Failed to read %s: %s", path, exc)
            continue
        combined = f"{path.stem}\n{content[:4000]}"
        documents.append(
            CorpusDocument(
                path=path,
                title=path.stem.replace("_", " "),
                content=content,
                tokens=_tokenize(combined),
                years=set(YEAR_PATTERN.findall(combined)),
            )
        )

    logger.info("Loaded %s corpus documents from %s", len(documents), corpus_dir)
    return documents
