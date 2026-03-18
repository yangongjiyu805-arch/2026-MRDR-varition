import os
import re
from dataclasses import dataclass
from datetime import datetime
import importlib
from pathlib import Path

os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import pandas as pd
import scanpy as sc


AGGREGATE_COLUMNS = [
    ("Total", "Total"),
    ("Bio conservation", "Bio conservation"),
    ("Batch correction", "Batch correction"),
    ("Modality integration", "Modality integration"),
]

CORE_RAW_COLUMNS = [
    ("silhouette_label", "Silhouette label"),
    ("silhouette_batch_b", "Silhouette batch"),
    ("silhouette_batch_m", "Silhouette modality"),
    ("ilisi_knn_b", "iLISI batch"),
    ("ilisi_knn_m", "iLISI modality"),
    ("graph_connectivity", "Graph connectivity"),
]


@dataclass(frozen=True)
class EmbeddingSpec:
    mode: str
    keys: tuple[str, ...]
    label: str
    expression: str


def _load_metric_modules():
    for module_name in ("scMIGRA.metrics_definitions", "metrics_definitions"):
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            short_name = module_name.rsplit(".", 1)[-1]
            if exc.name not in {module_name, short_name}:
                raise ModuleNotFoundError(
                    f"Missing dependency {exc.name!r} required for evaluation. "
                    "Please install scib-metrics in the runtime environment before running metrics.py."
                ) from exc
            continue
        return (
            module.BatchCorrection2,
            module.Benchmarker2,
            module.BioConservation2,
            module.ModalityIntegration2,
        )

    raise ModuleNotFoundError(
        "Could not import metrics_definitions. "
        "Please run metrics.py from the project environment where scMIGRA is available."
    )


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(value).strip()


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid bool env var: {name}={value}")


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def _parse_modality_map(text: str | None) -> dict[str, str]:
    if text is None or text.strip() == "":
        return {"0": "RNA", "1": "ATAC"}

    mapping: dict[str, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid modality mapping entry: {item!r}. Use key=value,key=value.")
        source, target = item.split("=", 1)
        mapping[source.strip()] = target.strip()
    return mapping


def _infer_label_key(adata, preferred: str | None) -> str:
    candidates: list[str] = []
    if preferred is not None:
        candidates.append(preferred)
    candidates.extend(["cell_type", "celltype", "CellType", "label", "labels"])

    checked: list[str] = []
    for candidate in candidates:
        if candidate not in checked:
            checked.append(candidate)
        if candidate in adata.obs.columns:
            return candidate

    raise KeyError(
        "Could not find a label key for evaluation. "
        f"Tried {checked}, available obs columns are {list(adata.obs.columns)}."
    )


def _require_obs_column(adata, key: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(f"Required obs column {key!r} not found in evaluation h5ad.")


def _require_obsm_key(adata, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError(f"Required obsm key {key!r} not found in evaluation h5ad.")


def _to_dense_array(value) -> np.ndarray:
    if hasattr(value, "toarray"):
        value = value.toarray()
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D embedding array, got shape {array.shape}.")
    return array.astype(np.float32, copy=False)


def _zscore_block(block: np.ndarray) -> np.ndarray:
    mean = block.mean(axis=0, keepdims=True)
    std = block.std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return (block - mean) / std


def _slugify(text: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z_]+", "_", text.strip())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "embedding"


def _default_embedding_specs(adata) -> list[EmbeddingSpec]:
    specs: list[EmbeddingSpec] = []

    if "latent_shared" in adata.obsm:
        specs.append(
            EmbeddingSpec(
                mode="single",
                keys=("latent_shared",),
                label="Shared",
                expression="latent_shared",
            )
        )

    if "latent_shared" in adata.obsm and "latent_specific" in adata.obsm:
        specs.append(
            EmbeddingSpec(
                mode="concat",
                keys=("latent_shared", "latent_specific"),
                label="SharedSpecific",
                expression="concat:latent_shared+latent_specific",
            )
        )

    if "program_assignments" in adata.obsm:
        specs.append(
            EmbeddingSpec(
                mode="single",
                keys=("program_assignments",),
                label="Programs",
                expression="program_assignments",
            )
        )

    if specs:
        return specs

    fallback_keys: list[str] = []
    for key in adata.obsm.keys():
        try:
            _to_dense_array(adata.obsm[key])
        except Exception:
            continue
        fallback_keys.append(key)

    if not fallback_keys:
        raise KeyError("No usable obsm embedding was found for evaluation.")

    first_key = fallback_keys[0]
    return [
        EmbeddingSpec(
            mode="single",
            keys=(first_key,),
            label=first_key,
            expression=first_key,
        )
    ]


def _parse_embedding_specs(text: str | None, adata) -> list[EmbeddingSpec]:
    if text is None or text.strip() == "":
        return _default_embedding_specs(adata)

    specs: list[EmbeddingSpec] = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue

        if "=" in item:
            expression, label = item.split("=", 1)
            expression = expression.strip()
            label = label.strip()
        else:
            expression = item
            label = item

        if expression.startswith("concat:"):
            keys = tuple(part.strip() for part in expression[len("concat:") :].split("+") if part.strip())
            if len(keys) < 2:
                raise ValueError(f"Invalid concat embedding spec: {item!r}")
            specs.append(EmbeddingSpec(mode="concat", keys=keys, label=label, expression=expression))
        else:
            specs.append(EmbeddingSpec(mode="single", keys=(expression,), label=label, expression=expression))

    if not specs:
        raise ValueError("No embedding specs were parsed from MRDR_METRICS_EMBEDDINGS.")
    return specs


def _build_embedding_array(adata, spec: EmbeddingSpec) -> np.ndarray:
    if spec.mode == "single":
        _require_obsm_key(adata, spec.keys[0])
        return _to_dense_array(adata.obsm[spec.keys[0]]).copy()

    if spec.mode == "concat":
        blocks: list[np.ndarray] = []
        for key in spec.keys:
            _require_obsm_key(adata, key)
            block = _to_dense_array(adata.obsm[key])
            # 中文注释：
            # 不同 latent 分支的数值尺度常常不同，直接拼接会让高方差分支主导邻居图。
            # 这里先按列标准化，再做拼接，更适合高频实验时稳定比较不同组合。
            blocks.append(_zscore_block(block))
        return np.concatenate(blocks, axis=1).astype(np.float32, copy=False)

    raise ValueError(f"Unsupported embedding mode: {spec.mode}")


def _numeric_value(df: pd.DataFrame, row_key: str, column_key: str) -> float:
    return float(pd.to_numeric(pd.Series([df.loc[row_key, column_key]]), errors="raise").iloc[0])


def _make_method_name(prefix: str | None, label: str) -> str:
    if prefix is None or prefix.strip() == "":
        return label
    return f"{prefix}::{label}"


def _history_path(output_dir: Path, record_file: str | None) -> Path | None:
    if record_file is None or record_file.strip() == "":
        return None
    path = Path(record_file)
    if not path.is_absolute():
        path = output_dir / path
    return path


def _build_summary(
    raw_results: pd.DataFrame,
    scaled_results: pd.DataFrame,
    evaluated_methods: list[dict[str, str]],
    tag: str,
    input_h5ad: Path,
    label_key: str,
    batch_key: str,
    modality_key: str,
) -> pd.DataFrame:
    now_text = datetime.now().astimezone().isoformat(timespec="seconds")
    has_multi_embeddings = len(evaluated_methods) > 1

    rows: list[dict[str, object]] = []
    for method in evaluated_methods:
        method_name = method["method_name"]
        row: dict[str, object] = {
            "Timestamp": now_text,
            "Tag": tag,
            "InputH5AD": str(input_h5ad),
            "Method": method_name,
            "EmbeddingLabel": method["label"],
            "EmbeddingSpec": method["expression"],
            "BatchKey": batch_key,
            "ModalityKey": modality_key,
            "LabelKey": label_key,
        }

        for source_col, output_col in AGGREGATE_COLUMNS:
            row[output_col] = _numeric_value(raw_results, method_name, source_col)

        if has_multi_embeddings:
            for source_col, output_col in AGGREGATE_COLUMNS:
                row[f"{output_col} scaled"] = _numeric_value(scaled_results, method_name, source_col)
        else:
            for _, output_col in AGGREGATE_COLUMNS:
                row[f"{output_col} scaled"] = np.nan

        for source_col, output_col in CORE_RAW_COLUMNS:
            row[output_col] = _numeric_value(raw_results, method_name, source_col)

        rows.append(row)

    summary = pd.DataFrame(rows)
    if has_multi_embeddings:
        summary = summary.sort_values(by=["Total scaled", "Total"], ascending=[False, False], kind="mergesort")
    else:
        summary = summary.sort_values(by=["Total"], ascending=[False], kind="mergesort")
    summary.insert(0, "WithinRunRank", np.arange(1, len(summary) + 1))
    return summary.reset_index(drop=True)


def _append_history(summary: pd.DataFrame, history_path: Path | None) -> None:
    if history_path is None:
        return

    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        history_df = pd.read_csv(history_path)
        combined = pd.concat([history_df, summary], ignore_index=True, sort=False)
    else:
        combined = summary.copy()
    combined.to_csv(history_path, index=False)


def _print_summary(summary: pd.DataFrame) -> None:
    print("[MIGRA_METRICS] Quick summary")
    for _, row in summary.iterrows():
        line = (
            f"[MIGRA_METRICS] rank={int(row['WithinRunRank'])} "
            f"method={row['Method']} "
            f"Total={row['Total']:.4f} "
            f"Bio={row['Bio conservation']:.4f} "
            f"Batch={row['Batch correction']:.4f} "
            f"Modality={row['Modality integration']:.4f}"
        )
        if not pd.isna(row["Total scaled"]):
            line += f" TotalScaled={row['Total scaled']:.4f}"
        print(line)


def main() -> None:
    # 中文实验流程注释：
    # 1. 读入训练后保存的 h5ad。
    # 2. 自动准备一个或多个待评估 embedding，默认优先检查 shared / shared+specific / programs。
    # 3. 输出 summary、full raw、full scaled、history 四类结果，方便反复做实验时直接横向比较。
    BatchCorrection2, Benchmarker2, BioConservation2, ModalityIntegration2 = _load_metric_modules()

    input_h5ad = Path(_require_env("MRDR_METRICS_INPUT_H5AD")).resolve()
    output_dir_env = _get_env("MRDR_METRICS_OUTPUT_DIR")
    output_dir = Path(output_dir_env).resolve() if output_dir_env else (input_h5ad.parent / "metrics").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_jobs = _get_env_int("MRDR_METRICS_NJOBS", 1)
    batch_key = _get_env("MRDR_METRICS_BATCH_KEY", "batch")
    modality_key = _get_env("MRDR_METRICS_MODALITY_KEY", "modality")
    label_key_hint = _get_env("MRDR_METRICS_LABEL_KEY")
    tag = _get_env("MRDR_METRICS_TAG", input_h5ad.stem)
    method_prefix = _get_env("MRDR_METRICS_METHOD_PREFIX", input_h5ad.stem)
    record_file = _get_env("MRDR_METRICS_RECORD_FILE", "metrics_history.csv")
    plot_tables = _get_env_bool("MRDR_METRICS_PLOT", False)
    modality_map = _parse_modality_map(_get_env("MRDR_METRICS_MODALITY_MAP", "0=RNA,1=ATAC"))

    adata = sc.read_h5ad(str(input_h5ad))
    _require_obs_column(adata, batch_key)
    _require_obs_column(adata, modality_key)

    label_key = _infer_label_key(adata, label_key_hint)
    embedding_specs = _parse_embedding_specs(_get_env("MRDR_METRICS_EMBEDDINGS"), adata)

    adata.obs[modality_key] = adata.obs[modality_key].astype(str)
    for source_name, target_name in modality_map.items():
        adata.obs.loc[adata.obs[modality_key] == source_name, modality_key] = target_name

    evaluated_methods: list[dict[str, str]] = []
    obsm_keys: list[str] = []
    for index, spec in enumerate(embedding_specs, start=1):
        eval_key = f"__eval__{index:02d}_{_slugify(spec.label)}"
        method_name = _make_method_name(method_prefix, spec.label)
        adata.obsm[eval_key] = _build_embedding_array(adata, spec)
        obsm_keys.append(eval_key)
        evaluated_methods.append(
            {
                "method_name": eval_key,
                "display_name": method_name,
                "label": spec.label,
                "expression": spec.expression,
            }
        )

    benchmarker = Benchmarker2(
        adata,
        batch_key=batch_key,
        label_key=label_key,
        modality_key=modality_key,
        bio_conservation_metrics=BioConservation2(),
        batch_correction_metrics=BatchCorrection2(),
        modality_integration_metrics=ModalityIntegration2(),
        embedding_obsm_keys=obsm_keys,
        n_jobs=n_jobs,
    )
    benchmarker.benchmark()

    raw_results = benchmarker.get_results(min_max_scale=False).rename(
        index={item["method_name"]: item["display_name"] for item in evaluated_methods}
    )
    scaled_results = benchmarker.get_results(min_max_scale=True).rename(
        index={item["method_name"]: item["display_name"] for item in evaluated_methods}
    )
    if len(embedding_specs) == 1:
        scaled_results = raw_results.copy()

    summary = _build_summary(
        raw_results=raw_results,
        scaled_results=scaled_results,
        evaluated_methods=[
            {
                "method_name": item["display_name"],
                "label": item["label"],
                "expression": item["expression"],
            }
            for item in evaluated_methods
        ],
        tag=tag,
        input_h5ad=input_h5ad,
        label_key=label_key,
        batch_key=batch_key,
        modality_key=modality_key,
    )

    summary_path = output_dir / f"summary_metrics_{tag}.csv"
    raw_path = output_dir / f"full_metrics_raw_{tag}.csv"
    scaled_path = output_dir / f"full_metrics_scaled_{tag}.csv"
    history_path = _history_path(output_dir, record_file)

    summary.to_csv(summary_path, index=False)
    raw_results.to_csv(raw_path)
    scaled_results.to_csv(scaled_path)
    _append_history(summary, history_path)

    if plot_tables:
        benchmarker.plot_results_table(tag=f"raw_table_{tag}", min_max_scale=False, show=False, save_dir=str(output_dir))
        benchmarker.plot_results_table(tag=f"scaled_table_{tag}", min_max_scale=True, show=False, save_dir=str(output_dir))

    print(f"[MIGRA_METRICS] label_key={label_key}")
    print(f"[MIGRA_METRICS] evaluated_embeddings={len(embedding_specs)}")
    if len(embedding_specs) == 1:
        print("[MIGRA_METRICS] only one embedding was evaluated; scaled aggregate scores are kept as NaN in summary.")
    _print_summary(summary)
    print(f"[MIGRA_METRICS] summary saved to {summary_path}")
    print(f"[MIGRA_METRICS] raw metrics saved to {raw_path}")
    print(f"[MIGRA_METRICS] scaled metrics saved to {scaled_path}")
    if history_path is not None:
        print(f"[MIGRA_METRICS] history updated at {history_path}")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1) from None
