import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Any
import re

import matplotlib as mpl
import matplotlib.pyplot as plt

# 永久解决服务器找不到 Arial 字体警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

import scib_metrics
from scib_metrics.nearest_neighbors import NeighborsResults, pynndescent

try:
    from plottable import ColumnDefinition, Table
    from plottable.cmap import normed_cmap
    from plottable.plots import bar

    _PLOTTABLE_AVAILABLE = True
except ModuleNotFoundError:
    ColumnDefinition = None
    Table = None
    normed_cmap = None
    bar = None
    _PLOTTABLE_AVAILABLE = False

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.family"] = "DejaVu Sans"

Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_LABELS = "labels"
_BATCH = "batch"
_MODALITY = "modality"
_X_PRE = "X_pre"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"
_METRIC_NAME = "Metric Name"


@dataclass(frozen=True)
class BioConservation2:
    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = False
    nmi_ari_cluster_labels_kmeans: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True


@dataclass(frozen=True)
class BatchCorrection2:
    silhouette_batch_b: MetricType = True
    ilisi_knn_b: MetricType = True
    kbet_per_label_b: MetricType = True
    pcr_comparison_b: MetricType = True


@dataclass(frozen=True)
class ModalityIntegration2:
    silhouette_batch_m: MetricType = True
    ilisi_knn_m: MetricType = True
    kbet_per_label_m: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison_m: MetricType = True


metric_name_cleaner2 = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch_b": "Silhouette batch",
    "silhouette_batch_m": "Silhouette modality",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn_b": "iLISI",
    "ilisi_knn_m": "iLISI",
    "kbet_per_label_b": "KBET",
    "kbet_per_label_m": "KBET",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison_b": "PCR comparison",
    "pcr_comparison_m": "PCR comparison",
}


class MetricAnnDataAPI2(Enum):
    isolated_labels = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_b = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison_b = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn_b = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label_b = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    graph_connectivity = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    silhouette_batch_m = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_MODALITY])
    pcr_comparison_m = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_MODALITY], categorical=True)
    ilisi_knn_m = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_MODALITY])
    kbet_per_label_m = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_MODALITY], ad.obs[_LABELS])


class Benchmarker2:
    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        modality_key: str,
        embedding_obsm_keys: list[str],
        bio_conservation_metrics: BioConservation2 | None,
        batch_correction_metrics: BatchCorrection2 | None,
        modality_integration_metrics: ModalityIntegration2 | None,
        pre_integrated_embedding_obsm_key: str | None = None,
        n_jobs: int = 1,
        progress_bar: bool = True,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._modality_integration_metrics = modality_integration_metrics
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._modality_key = modality_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        if self._bio_conservation_metrics is None and self._batch_correction_metrics is None:
            raise ValueError("Either batch or bio metrics must be defined.")
        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict.update({"Bio conservation": self._bio_conservation_metrics})
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict.update({"Batch correction": self._batch_correction_metrics})
        if self._modality_integration_metrics is not None:
            self._metric_collection_dict.update({"Modality integration": self._modality_integration_metrics})

    def prepare(self, neighbor_computer: Callable[[np.ndarray, int], NeighborsResults] | None = None) -> None:
        if self._pre_integrated_embedding_obsm_key is None:
            sc.tl.pca(self._adata, use_highly_variable=False)
            self._pre_integrated_embedding_obsm_key = "X_pca"
        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_MODALITY] = np.asarray(self._adata.obs[self._modality_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]
        progress = self._emb_adatas.values()
        if self._progress_bar:
            progress = tqdm(progress, desc="Computing neighbors")
        for ad in progress:
            if neighbor_computer is not None:
                neigh_result = neighbor_computer(ad.X, max(self._neighbor_values))
            else:
                neigh_result = pynndescent(ad.X, n_neighbors=max(self._neighbor_values), random_state=0, n_jobs=self._n_jobs)
            for n in self._neighbor_values:
                ad.uns[f"{n}_neighbor_res"] = neigh_result.subset_neighbors(n=n)
        self._prepared = True

    def benchmark(self) -> None:
        if self._benchmarked:
            warnings.warn(
                "The benchmark has already been run. Running it again will overwrite the previous results.",
                UserWarning,
            )
        if not self._prepared:
            self.prepare()
        num_metrics = sum([sum([v is not False for v in asdict(met_col)]) for met_col in self._metric_collection_dict.values()])
        progress_embs = self._emb_adatas.items()
        if self._progress_bar:
            progress_embs = tqdm(self._emb_adatas.items(), desc="Embeddings", position=0, colour="green")
        for emb_key, ad in progress_embs:
            pbar = None
            if self._progress_bar:
                pbar = tqdm(total=num_metrics, desc="Metrics", position=1, leave=False, colour="blue")
            for metric_type, metric_collection in self._metric_collection_dict.items():
                for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                    if use_metric_or_kwargs:
                        if pbar is not None:
                            pbar.set_postfix_str(f"{metric_type}: {metric_name}")
                        metric_fn = getattr(scib_metrics, re.sub(r"(_b|_m)$", "", metric_name))
                        if isinstance(use_metric_or_kwargs, dict):
                            metric_fn = partial(metric_fn, **use_metric_or_kwargs)
                        metric_value = getattr(MetricAnnDataAPI2, metric_name)(ad, metric_fn)
                        if isinstance(metric_value, dict):
                            for k, v in metric_value.items():
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", emb_key] = v
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_TYPE] = metric_type
                                self._results.loc[f"{metric_type}_{metric_name}_{k}", _METRIC_NAME] = f"{metric_name}_{k}"
                        else:
                            self._results.loc[f"{metric_type}_{metric_name}", emb_key] = metric_value
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_TYPE] = metric_type
                            self._results.loc[f"{metric_type}_{metric_name}", _METRIC_NAME] = metric_name
                        if pbar is not None:
                            pbar.update(1)
        self._benchmarked = True

    def get_results(self, min_max_scale: bool = True, clean_names: bool = True) -> pd.DataFrame:
        df = self._results.transpose()
        df.index.name = "Embedding"
        df = df.loc[~df.index.isin([_METRIC_TYPE, _METRIC_NAME])]
        if min_max_scale:
            df = pd.DataFrame(
                MinMaxScaler().fit_transform(df),
                columns=self._results[_METRIC_NAME].values,
                index=df.index,
            )
        else:
            df = pd.DataFrame(
                df.to_numpy(),
                columns=self._results[_METRIC_NAME].values,
                index=df.index,
            )
        df = df.transpose()
        df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values
        per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
        if (
            self._modality_integration_metrics is not None
            and self._batch_correction_metrics is not None
            and self._bio_conservation_metrics is not None
        ):
            per_class_score["Total"] = (
                0.3 * per_class_score["Batch correction"]
                + 0.3 * per_class_score["Modality integration"]
                + 0.4 * per_class_score["Bio conservation"]
            )
        elif (
            self._modality_integration_metrics is not None
            and self._bio_conservation_metrics is not None
            and self._batch_correction_metrics is None
        ):
            per_class_score["Total"] = (
                0.4 * per_class_score["Modality integration"] + 0.6 * per_class_score["Bio conservation"]
            )
        df[_METRIC_NAME] = self._results[_METRIC_NAME].values
        df = pd.concat([df.transpose(), per_class_score], axis=1)
        # Pandas 2.x enforces strict column dtypes; cast aggregate columns to object
        # before writing string metadata rows (_METRIC_TYPE/_METRIC_NAME).
        df[per_class_score.columns] = df[per_class_score.columns].astype(object)
        df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        df.loc[_METRIC_NAME, per_class_score.columns] = per_class_score.columns
        return df

    def plot_results_table(self, tag, min_max_scale: bool = True, show: bool = True, save_dir: str | None = None):
        if not _PLOTTABLE_AVAILABLE:
            warnings.warn(
                "plottable is not installed; skip PDF table plotting. CSV metrics are still available.",
                UserWarning,
            )
            return None
        num_embeds = len(self._embedding_obsm_keys)
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=mpl.cm.PRGn, num_stds=2.5)
        df = self.get_results(min_max_scale=min_max_scale)
        plot_df = df.drop([_METRIC_TYPE, _METRIC_NAME], axis=0)
        if self._batch_correction_metrics is not None and self._bio_conservation_metrics is not None:
            sort_col = "Total"
        elif self._modality_integration_metrics is not None:
            sort_col = "Modality integration"
        elif self._batch_correction_metrics is not None:
            sort_col = "Batch correction"
        else:
            sort_col = "Bio conservation"
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = df.columns[df.loc[_METRIC_TYPE] != _AGGREGATE_SCORE]
        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                title=metric_name_cleaner2[df.loc[_METRIC_NAME, col]].replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(plot_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for col in other_cols
        ]
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=df.loc[_METRIC_NAME, col].replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": mpl.cm.YlGnBu,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        with mpl.rc_context({"svg.fonttype": "none"}):
            fig, ax = plt.subplots(figsize=(len(df.columns) * 1.25, 3 + 0.3 * num_embeds))
            tab = Table(
                plot_df,
                cell_kw={
                    "linewidth": 0,
                    "edgecolor": "k",
                },
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": 10, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            fig.savefig(
                os.path.join(save_dir, f"{tag}.pdf"),
                facecolor=ax.get_facecolor(),
                dpi=300,
                format="pdf",
                bbox_inches="tight",
            )
        return tab
