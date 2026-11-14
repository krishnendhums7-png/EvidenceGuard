"""Offline Vector Index Building Script for EvidenceGuard.

Ingests Wikipedia dump records, cleans and normalizes sentences, extracts metadata,
computes dense vector representations via Sentence Transformers, and constructs
a persistent FAISS index for local semantic retrieval.

Usage:
    # Build index from sample dataset:
    python scripts/build_index.py --mode sample

    # Build index from a raw Wikipedia corpus file with a page limit:
    python scripts/build_index.py --mode custom --input data/raw/wiki-001.jsonl --max-pages 1000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import config
from src.corpus_loader import CorpusLoader
from src.embeddings import EmbeddingService, get_embedding_service
from src.logger import get_logger
from src.preprocess import EvidencePreprocessor
from src.retriever import SemanticRetriever
from src.vector_store import FAISSVectorStore

logger = get_logger("build_index")


def build_evidence_index(
    input_file: Path,
    output_dir: Path,
    index_name: str = "evidence_faiss",
    batch_size: int = 64,
    max_pages: int | None = None,
    validate: bool = True,
) -> None:
    """Execute full offline evidence indexing pipeline."""
    start_time = time.perf_counter()
    print("=" * 80)
    print("EVIDENCEGUARD: FAISS VECTOR INDEX GENERATION PIPELINE")
    print("=" * 80)
    print(f"Input Source:       {input_file}")
    print(f"Index Output Dir:   {output_dir}")
    print(f"Index Identifier:   {index_name}")
    print(f"Batch Size:         {batch_size}")
    print(f"Max Pages to Ingest:{max_pages or 'All available'}")
    print("-" * 80)

    # 1. Corpus Ingestion
    logger.info("Step 1/5: Streaming raw Wikipedia pages from %s ...", input_file)
    pages_generator = CorpusLoader.load_wikipedia_dump(input_file, max_pages=max_pages)

    # 2. Preprocessing & Sentence Extraction
    logger.info("Step 2/5: Extracting and normalizing evidence sentences ...")
    preprocessor = EvidencePreprocessor()
    evidence_records = list(preprocessor.process_corpus(pages_generator, deduplicate=True))
    total_records = len(evidence_records)

    if total_records == 0:
        logger.error("No valid evidence sentences extracted. Aborting index creation.")
        sys.exit(1)

    print(f"Extracted {total_records} distinct, normalized evidence sentences.")

    # 3. Dense Vector Embedding Generation
    logger.info("Step 3/5: Generating dense embeddings using Sentence Transformers ...")
    embedder = get_embedding_service()
    texts_to_embed = [rec.evidence_text for rec in evidence_records]

    embed_start = time.perf_counter()
    embeddings = embedder.embed_texts(
        texts=texts_to_embed,
        batch_size=batch_size,
        normalize=True,
        show_progress=True,
    )
    embed_elapsed = time.perf_counter() - embed_start
    print(
        f"Generated {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]} "
        f"in {embed_elapsed:.2f}s ({embeddings.shape[0]/max(embed_elapsed, 0.001):.1f} sent/sec)."
    )

    # 4. FAISS Index Construction & Persistence
    logger.info("Step 4/5: Building FAISS vector store and writing to disk ...")
    vector_store = FAISSVectorStore(dimension=embedder.dimension)
    metadata_list = [rec.to_dict() for rec in evidence_records]

    vector_store.add_vectors(vectors=embeddings, metadata_records=metadata_list)
    index_file, meta_file = vector_store.save(output_dir=output_dir, index_name=index_name)

    print(f"FAISS index binary written: {index_file} ({index_file.stat().st_size / 1024:.1f} KB)")
    print(f"Metadata written:           {meta_file} ({meta_file.stat().st_size / 1024:.1f} KB)")

    # 5. Validation Check
    if validate:
        logger.info("Step 5/5: Validating index reload and performing test retrieval ...")
        reloaded_store = FAISSVectorStore(dimension=embedder.dimension)
        reloaded_store.load(index_dir=output_dir, index_name=index_name)

        if reloaded_store.total_vectors != total_records:
            logger.error("Index validation failed! Count mismatch after reload.")
            sys.exit(1)

        retriever = SemanticRetriever(vector_store=reloaded_store, embedding_service=embedder)

        # Test query with known factual ground truth
        test_queries = [
            "Albert Einstein was a theoretical physicist born in Germany.",
            "Marie Curie was the first woman to win a Nobel Prize.",
            "The Pacific Ocean is the largest ocean on Earth.",
        ]

        print("\n" + "-" * 80)
        print("SAMPLE RETRIEVAL VALIDATION RESULTS:")
        print("-" * 80)
        for query in test_queries:
            res = retriever.retrieve(query, top_k=2)
            print(f"\nQuery: '{query}' (Retrieved {res.total_retrieved} in {res.retrieval_time_sec*1000:.1f}ms)")
            for i, p in enumerate(res.passages, start=1):
                print(f"  [{i}] (Score: {p.similarity_score:.4f}) [{p.human_title} §{p.sentence_id}]: {p.evidence_text}")

    total_time = time.perf_counter() - start_time
    print("=" * 80)
    print(f"INDEX GENERATION COMPLETED SUCCESSFULLY in {total_time:.2f} seconds.")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="EvidenceGuard FAISS Index Builder")
    parser.add_argument(
        "--mode",
        choices=["sample", "custom"],
        default="sample",
        help="Build mode: 'sample' uses the verified sample dataset; 'custom' uses a custom file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input Wikipedia dump JSONL (required if --mode custom).",
    )
    parser.add_argument(
        "--index-name",
        type=str,
        default="evidence_faiss",
        help="Base name for the generated index files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(config.indexes_dir),
        help="Target directory for FAISS index and metadata files.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Inference batch size for embedding generation.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Maximum Wikipedia pages to process (optional).",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Disable automatic validation retrieval queries after build.",
    )

    args = parser.parse_args()

    if args.mode == "sample":
        input_path = config.samples_dir / "wikipedia_sample.jsonl"
    else:
        if not args.input:
            print("Error: --input is required when --mode custom is chosen.")
            sys.exit(1)
        input_path = Path(args.input)

    if not input_path.is_file():
        print(f"Error: Target input file not found: {input_path}")
        sys.exit(1)

    build_evidence_index(
        input_file=input_path,
        output_dir=Path(args.output_dir),
        index_name=args.index_name,
        batch_size=args.batch_size,
        max_pages=args.max_pages,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()
