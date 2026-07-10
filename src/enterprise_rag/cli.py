from __future__ import annotations

import argparse
import time
from pathlib import Path

from enterprise_rag.cache.base import CacheStore
from enterprise_rag.cache.factory import create_cache
from enterprise_rag.config import load_config
from enterprise_rag.evaluation.eval_generation import (
    generate_eval_cases_from_logs,
    promote_reviewed_eval_draft,
    write_generated_eval_cases,
)
from enterprise_rag.evaluation.evidence_suggestion import approve_suggested_evidence, suggest_evidence_for_eval_draft
from enterprise_rag.evaluation.experiments import format_retrieval_experiment_report, run_top_k_experiments
from enterprise_rag.evaluation.index_inspection import format_index_quality_report, inspect_index
from enterprise_rag.evaluation.readiness import build_readiness_report, format_readiness_report
from enterprise_rag.evaluation.retrieval_eval import (
    format_retrieval_eval_report,
    load_retrieval_eval_cases,
    run_retrieval_eval,
)
from enterprise_rag.evaluation.self_healing_workflow import (
    format_self_healing_workflow_report,
    run_self_healing_workflow,
)
from enterprise_rag.indexing.vector_sync import VectorIndexSync
from enterprise_rag.ingestion.pipeline import IncrementalIngestPipeline
from enterprise_rag.ingestion.policy import IngestionFilePolicy
from enterprise_rag.jobs.ingest_jobs import JsonIngestJobStore
from enterprise_rag.jobs.runner import IngestJobRunner
from enterprise_rag.leases.factory import create_lease_store
from enterprise_rag.observability.log_analysis import analyze_query_log, format_log_analysis_report
from enterprise_rag.observability.query_logging import QueryLogger, build_query_log_record
from enterprise_rag.observability.tracing import format_query_trace
from enterprise_rag.rag.citations import CitationFormatter
from enterprise_rag.rag.pipeline import RagPipeline
from enterprise_rag.storage.json_store import JsonChunkStore
from enterprise_rag.vector_index.base import VectorIndex
from enterprise_rag.vector_index.factory import create_vector_index

DEFAULT_INDEX = Path("data/processed/chunks.json")
DEFAULT_JOBS = Path("data/jobs/ingest_jobs.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enterprise RAG local demo")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Load, clean, parse, and chunk documents")
    ingest_parser.add_argument("path", type=Path, help="File or directory to ingest")
    ingest_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ingest_parser.add_argument("--config", type=Path, help="JSON config file for vector index settings")
    ingest_parser.add_argument("--sync-vectors", action="store_true", help="Sync changed chunks to the vector index")
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview ingest changes without writing the index",
    )
    ingest_parser.add_argument("--allowed-group", action="append", default=None, help="Allowed group for ingested docs")

    run_job_parser = subparsers.add_parser("run-job", help="Run one queued ingest job from the persistent job store")
    run_job_parser.add_argument("job_id")
    run_job_parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    run_job_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    run_job_parser.add_argument("--config", type=Path, help="JSON config file for vector index settings")

    worker_parser = subparsers.add_parser("worker", help="Poll the persistent job store and run ingest jobs")
    worker_parser.add_argument("--jobs", type=Path, default=DEFAULT_JOBS)
    worker_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    worker_parser.add_argument("--config", type=Path, help="JSON config file for worker settings")
    worker_parser.add_argument("--once", action="store_true", help="Run one polling pass and exit")
    worker_parser.add_argument("--poll-seconds", type=float, default=None)

    query_parser = subparsers.add_parser("query", help="Query the local RAG index")
    query_parser.add_argument("query")
    query_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    query_parser.add_argument("--config", type=Path, help="JSON config file for retrieval defaults")
    query_parser.add_argument("--top-k", type=int, default=None)
    query_parser.add_argument(
        "--enable-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable knowledge graph retrieval",
    )
    query_parser.add_argument("--graph-max-hops", type=int, default=None)
    query_parser.add_argument("--user-group", action="append", default=None, help="User group for ACL filtering")
    query_parser.add_argument("--trace", action="store_true", help="Print retrieval trace details")
    query_parser.add_argument("--log-query", type=Path, help="Append query summary to a JSONL log file")

    eval_parser = subparsers.add_parser("eval", help="Run retrieval evaluation against the local RAG index")
    eval_parser.add_argument("eval_path", type=Path)
    eval_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    eval_parser.add_argument("--config", type=Path, help="JSON config file for retrieval defaults")
    eval_parser.add_argument("--k", type=int, default=None)
    eval_parser.add_argument(
        "--enable-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable knowledge graph retrieval",
    )
    eval_parser.add_argument("--graph-max-hops", type=int, default=None)

    experiment_parser = subparsers.add_parser("experiment", help="Run retrieval self-healing experiments")
    experiment_parser.add_argument("eval_path", type=Path)
    experiment_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    experiment_parser.add_argument("--config", type=Path, help="JSON config file for retrieval defaults")
    experiment_parser.add_argument("--k-values", type=int, nargs="+", default=None)
    experiment_parser.add_argument(
        "--enable-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable knowledge graph retrieval",
    )
    experiment_parser.add_argument("--graph-max-hops", type=int, default=None)

    inspect_parser = subparsers.add_parser("inspect-index", help="Inspect local chunk index quality")
    inspect_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)

    analyze_logs_parser = subparsers.add_parser("analyze-logs", help="Analyze query logs for RAG self-healing")
    analyze_logs_parser.add_argument("log_path", type=Path)

    generate_eval_parser = subparsers.add_parser(
        "generate-eval-from-logs",
        help="Generate draft retrieval eval cases from failed query logs",
    )
    generate_eval_parser.add_argument("log_path", type=Path)
    generate_eval_parser.add_argument("--output", type=Path, required=True)
    generate_eval_parser.add_argument("--limit", type=int, default=20)

    promote_eval_parser = subparsers.add_parser(
        "promote-eval-draft",
        help="Promote human-reviewed eval draft cases into a formal eval file",
    )
    promote_eval_parser.add_argument("draft_path", type=Path)
    promote_eval_parser.add_argument("--output", type=Path, required=True)

    suggest_evidence_parser = subparsers.add_parser(
        "suggest-evidence",
        help="Add retrieval-based evidence suggestions to eval draft cases",
    )
    suggest_evidence_parser.add_argument("draft_path", type=Path)
    suggest_evidence_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    suggest_evidence_parser.add_argument("--output", type=Path, required=True)
    suggest_evidence_parser.add_argument("--top-k", type=int, default=3)

    approve_evidence_parser = subparsers.add_parser(
        "approve-suggested-evidence",
        help="Approve one suggested evidence item by copying it into expected_text_contains",
    )
    approve_evidence_parser.add_argument("draft_path", type=Path)
    approve_evidence_parser.add_argument("--case-id", required=True)
    approve_evidence_parser.add_argument("--suggestion-index", type=int, required=True)
    approve_evidence_parser.add_argument("--output", type=Path, required=True)

    self_healing_parser = subparsers.add_parser(
        "self-healing-report",
        help="Run log analysis, eval draft generation, and evidence suggestion",
    )
    self_healing_parser.add_argument("log_path", type=Path)
    self_healing_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    self_healing_parser.add_argument("--workdir", type=Path, required=True)
    self_healing_parser.add_argument("--limit", type=int, default=20)
    self_healing_parser.add_argument("--suggestion-top-k", type=int, default=3)

    readiness_parser = subparsers.add_parser("readiness-report", help="Summarize local RAG system readiness")
    readiness_parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    readiness_parser.add_argument("--eval", type=Path, dest="eval_path")
    readiness_parser.add_argument("--query-log", type=Path)
    readiness_parser.add_argument("--self-healing-dir", type=Path)
    readiness_parser.add_argument("--config", type=Path, help="JSON config file for production readiness checks")
    readiness_parser.add_argument("--k", type=int, default=5)

    args = parser.parse_args()
    if args.command == "ingest":
        ingest(args.path, args.index, args.config, args.sync_vectors, tuple(args.allowed_group or ()), args.dry_run)
    elif args.command == "run-job":
        run_job(args.job_id, args.jobs, args.index, args.config)
    elif args.command == "worker":
        worker(args.jobs, args.index, args.config, args.once, args.poll_seconds)
    elif args.command == "query":
        config = load_config(args.config)
        top_k = args.top_k if args.top_k is not None else config.retrieval.top_k
        enable_graph = args.enable_graph if args.enable_graph is not None else config.retrieval.enable_graph
        graph_max_hops = args.graph_max_hops if args.graph_max_hops is not None else config.retrieval.graph_max_hops
        user_groups = args.user_group if args.user_group is not None else config.security.default_user_groups
        query(
            args.query,
            args.index,
            top_k,
            enable_graph,
            graph_max_hops,
            set(user_groups),
            args.trace,
            args.log_query,
            create_vector_index(config.vector_index),
            create_cache(config.cache),
            config.cache.embedding_ttl_seconds,
        )
    elif args.command == "eval":
        config = load_config(args.config)
        k = args.k if args.k is not None else config.retrieval.top_k
        enable_graph = args.enable_graph if args.enable_graph is not None else config.retrieval.enable_graph
        graph_max_hops = args.graph_max_hops if args.graph_max_hops is not None else config.retrieval.graph_max_hops
        eval_retrieval(args.eval_path, args.index, k, enable_graph, graph_max_hops)
    elif args.command == "experiment":
        config = load_config(args.config)
        k_values = args.k_values if args.k_values is not None else list(config.retrieval.experiment_k_values)
        enable_graph = args.enable_graph if args.enable_graph is not None else config.retrieval.enable_graph
        graph_max_hops = args.graph_max_hops if args.graph_max_hops is not None else config.retrieval.graph_max_hops
        experiment_retrieval(args.eval_path, args.index, k_values, enable_graph, graph_max_hops)
    elif args.command == "inspect-index":
        inspect_local_index(args.index)
    elif args.command == "analyze-logs":
        analyze_logs(args.log_path)
    elif args.command == "generate-eval-from-logs":
        generate_eval_from_logs(args.log_path, args.output, args.limit)
    elif args.command == "promote-eval-draft":
        promote_eval_draft(args.draft_path, args.output)
    elif args.command == "suggest-evidence":
        suggest_evidence(args.draft_path, args.index, args.output, args.top_k)
    elif args.command == "approve-suggested-evidence":
        approve_evidence(args.draft_path, args.case_id, args.suggestion_index, args.output)
    elif args.command == "self-healing-report":
        self_healing_report(args.log_path, args.index, args.workdir, args.limit, args.suggestion_top_k)
    elif args.command == "readiness-report":
        readiness_report(args.index, args.eval_path, args.query_log, args.self_healing_dir, args.config, args.k)


def ingest(
    path: Path,
    index_path: Path,
    config_path: Path | None = None,
    sync_vectors: bool = False,
    allowed_groups: tuple[str, ...] = (),
    dry_run: bool = False,
) -> None:
    config = load_config(config_path)
    embedding_cache = create_cache(config.cache)
    store = JsonChunkStore(index_path)
    metadata_overrides = {"allowed_groups": ",".join(allowed_groups)} if allowed_groups else None
    report = IncrementalIngestPipeline(file_policy=IngestionFilePolicy.from_config(config.ingestion)).run(
        path, store, metadata_overrides=metadata_overrides, dry_run=dry_run
    )
    if dry_run:
        print("Dry run: index was not written.")
    print(f"Indexed {report.chunks_indexed} chunks from {report.documents_loaded} documents into {index_path}")
    print(
        "Ingest report: "
        f"new={report.documents_new}, "
        f"updated={report.documents_updated}, "
        f"unchanged={report.documents_unchanged}, "
        f"deleted={report.documents_deleted}, "
        f"filtered={report.documents_filtered}"
    )
    if sync_vectors and not dry_run:
        chunks_by_id = {chunk.id: chunk for chunk in store.load()}
        chunks_to_upsert = [chunks_by_id[id] for id in report.chunks_upserted if id in chunks_by_id]
        sync_report = VectorIndexSync(
            embedding_cache=embedding_cache,
            embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
        ).sync(
            create_vector_index(config.vector_index),
            chunks_to_upsert=chunks_to_upsert,
            chunk_ids_to_delete=list(report.chunks_deleted),
        )
        print(f"Vector sync report: upserted={sync_report.vectors_upserted}, deleted={sync_report.vectors_deleted}")


def run_job(job_id: str, jobs_path: Path, index_path: Path, config_path: Path | None = None) -> None:
    config = load_config(config_path)
    job_store = JsonIngestJobStore(jobs_path)
    if job_store.get(job_id) is None:
        raise SystemExit(f"No ingest job found for {job_id} in {jobs_path}")

    runner = IngestJobRunner(
        job_store=job_store,
        index_path=index_path,
        config=config,
        embedding_cache=create_cache(config.cache),
        embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
        lease_store=create_lease_store(config.leases),
    )
    runner.run(job_id)
    job = job_store.get(job_id)
    if job is None:
        raise SystemExit(f"Ingest job disappeared while running: {job_id}")
    print(f"Job {job.job_id} {job.status}")
    if job.report is not None:
        print(
            "Ingest report: "
            f"new={job.report.documents_new}, "
            f"updated={job.report.documents_updated}, "
            f"unchanged={job.report.documents_unchanged}, "
            f"deleted={job.report.documents_deleted}, "
            f"filtered={job.report.documents_filtered}, "
            f"chunks={job.report.chunks_indexed}"
        )
    if job.vector_sync is not None:
        print(
            "Vector sync report: "
            f"upserted={job.vector_sync.get('vectors_upserted', 0)}, "
            f"deleted={job.vector_sync.get('vectors_deleted', 0)}"
        )
    if job.error is not None:
        print(f"Error: {job.error}")


def worker(
    jobs_path: Path,
    index_path: Path,
    config_path: Path | None = None,
    once: bool = False,
    poll_seconds: float | None = None,
) -> None:
    config = load_config(config_path)
    interval = poll_seconds if poll_seconds is not None else config.jobs.worker_poll_seconds
    job_store = JsonIngestJobStore(jobs_path)
    runner = IngestJobRunner(
        job_store=job_store,
        index_path=index_path,
        config=config,
        embedding_cache=create_cache(config.cache),
        embedding_ttl_seconds=config.cache.embedding_ttl_seconds,
        lease_store=create_lease_store(config.leases),
    )

    while True:
        jobs = job_store.list()
        for job in jobs:
            runner.run(job.job_id)
        print(f"Worker scanned {len(jobs)} jobs from {jobs_path}")
        if once:
            return
        time.sleep(interval)


def query(
    query_text: str,
    index_path: Path,
    top_k: int,
    enable_graph: bool = False,
    graph_max_hops: int = 2,
    user_groups: set[str] | None = None,
    show_trace: bool = False,
    log_query_path: Path | None = None,
    vector_index: VectorIndex | None = None,
    embedding_cache: CacheStore | None = None,
    embedding_ttl_seconds: int | None = 86_400,
) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")

    pipeline = RagPipeline(
        chunks,
        enable_graph=enable_graph,
        graph_max_hops=graph_max_hops,
        vector_index=vector_index,
        embedding_cache=embedding_cache,
        embedding_ttl_seconds=embedding_ttl_seconds,
    )
    answer, trace = pipeline.answer_for_user_with_trace(
        query_text,
        top_k=top_k,
        user_groups=user_groups,
    )
    print(answer.answer)
    print("\nQuery plan")
    print(f"- normalized: {answer.query_plan.normalized_query}")
    print(f"- rewrites: {', '.join(answer.query_plan.rewritten_queries)}")
    if answer.query_plan.corrections:
        print(f"- corrections: {answer.query_plan.corrections}")
    if answer.query_plan.metadata_filters:
        print(f"- metadata filters: {answer.query_plan.metadata_filters}")
    if answer.query_plan.ambiguity_notes:
        print(f"- ambiguity: {'; '.join(answer.query_plan.ambiguity_notes)}")

    print("\nCitations")
    formatter = CitationFormatter()
    for citation in formatter.format_many(answer.citations):
        print(citation)

    if show_trace:
        print("\n" + format_query_trace(trace))

    if log_query_path is not None:
        record = build_query_log_record(
            answer,
            trace,
            top_k=top_k,
            enable_graph=enable_graph,
            graph_max_hops=graph_max_hops,
            user_groups=user_groups,
        )
        QueryLogger(log_query_path).log(record)
        print(f"\nLogged query to {log_query_path}")


def eval_retrieval(
    eval_path: Path,
    index_path: Path,
    k: int,
    enable_graph: bool = False,
    graph_max_hops: int = 2,
) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")

    cases = load_retrieval_eval_cases(eval_path, chunks)
    report = run_retrieval_eval(cases, chunks, k=k, enable_graph=enable_graph, graph_max_hops=graph_max_hops)
    print(format_retrieval_eval_report(report))


def experiment_retrieval(
    eval_path: Path,
    index_path: Path,
    k_values: list[int],
    enable_graph: bool = False,
    graph_max_hops: int = 2,
) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")

    cases = load_retrieval_eval_cases(eval_path, chunks)
    report = run_top_k_experiments(
        cases,
        chunks,
        k_values=k_values,
        enable_graph=enable_graph,
        graph_max_hops=graph_max_hops,
    )
    print(format_retrieval_experiment_report(report))


def inspect_local_index(index_path: Path) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")

    print(format_index_quality_report(inspect_index(chunks)))


def analyze_logs(log_path: Path) -> None:
    print(format_log_analysis_report(analyze_query_log(log_path)))


def generate_eval_from_logs(log_path: Path, output_path: Path, limit: int) -> None:
    cases = generate_eval_cases_from_logs(log_path, limit=limit)
    write_generated_eval_cases(cases, output_path)
    print(f"Wrote {len(cases)} draft eval cases to {output_path}")


def promote_eval_draft(draft_path: Path, output_path: Path) -> None:
    report = promote_reviewed_eval_draft(draft_path, output_path)
    print(f"Promoted {report.promoted_count} reviewed eval cases to {output_path}")
    if report.skipped_ids:
        print(f"Skipped {report.skipped_count} unreviewed cases: {', '.join(report.skipped_ids)}")


def suggest_evidence(draft_path: Path, index_path: Path, output_path: Path, top_k: int) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")
    suggest_evidence_for_eval_draft(draft_path, chunks, output_path, top_k=top_k)
    print(f"Wrote evidence suggestions to {output_path}")


def approve_evidence(draft_path: Path, case_id: str, suggestion_index: int, output_path: Path) -> None:
    approve_suggested_evidence(draft_path, case_id, suggestion_index, output_path)
    print(f"Approved suggested evidence {suggestion_index} for {case_id} into {output_path}")


def self_healing_report(
    log_path: Path,
    index_path: Path,
    workdir: Path,
    limit: int,
    suggestion_top_k: int,
) -> None:
    chunks = JsonChunkStore(index_path).load()
    if not chunks:
        raise SystemExit(f"No chunks found at {index_path}. Run `enterprise-rag ingest data/raw` first.")
    report = run_self_healing_workflow(
        log_path,
        chunks,
        workdir,
        limit=limit,
        suggestion_top_k=suggestion_top_k,
    )
    print(format_self_healing_workflow_report(report))


def readiness_report(
    index_path: Path,
    eval_path: Path | None,
    query_log_path: Path | None,
    self_healing_dir: Path | None,
    config_path: Path | None,
    k: int,
) -> None:
    chunks = JsonChunkStore(index_path).load()
    config = load_config(config_path)
    report = build_readiness_report(
        chunks,
        index_path=index_path,
        eval_path=eval_path,
        query_log_path=query_log_path,
        self_healing_dir=self_healing_dir,
        config=config,
        k=k,
    )
    print(format_readiness_report(report, k=k))


if __name__ == "__main__":
    main()
