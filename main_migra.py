import os
import random
import sys
from datetime import datetime
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import torch


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return str(value).strip()


def _get_env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip()


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return int(str(value).strip())


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    return float(str(value).strip())


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


def _get_env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return None
    return int(str(value).strip())


def _get_env_int_list(name: str, default: list[int]) -> list[int]:
    value = os.environ.get(name)
    if value is None or str(value).strip() == "":
        return default
    text = str(value).replace(",", " ")
    items = [x for x in text.split() if x]
    if not items:
        return default
    return [int(x) for x in items]


def _require_obs_column(adata: ad.AnnData, key: str) -> None:
    if key not in adata.obs.columns:
        raise KeyError(f"Required obs column {key!r} not found in input h5ad.")


def _resolve_input_layer(adata: ad.AnnData, preferred: str | None) -> str | None:
    if preferred is not None and str(preferred).strip() != "":
        layer_name = str(preferred).strip()
        if layer_name.lower() in {"none", "x", "adata.x"}:
            return None
        if layer_name not in adata.layers:
            raise KeyError(
                f"Requested input layer {layer_name!r} was not found. "
                f"Available layers are {list(adata.layers.keys())}."
            )
        return layer_name

    for candidate in ["counts", "count", "raw_counts", "Counts"]:
        if candidate in adata.layers:
            return candidate
    return None


def _infer_existing_label_key(adata: ad.AnnData) -> str | None:
    for candidate in ["cell_type", "celltype", "CellType", "label", "labels"]:
        if candidate in adata.obs.columns:
            return candidate
    return None


def _set_global_seed(seed: int) -> None:
    # 中文说明：
    # 第一版先看方法方向是否成立，所以建议固定随机种子，避免不同轮次的随机波动掩盖模型本身差异。
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    src_dir = Path(_require_env("MRDR_SRC_DIR")).resolve()
    sys.path.insert(1, str(src_dir))

    from scMIGRA.module import Integration

    input_h5ad = Path(_require_env("MRDR_INPUT_H5AD")).resolve()
    output_h5ad = Path(_require_env("MRDR_OUTPUT_H5AD")).resolve()
    output_h5ad.parent.mkdir(parents=True, exist_ok=True)

    random_seed = _get_env_optional_int("MRDR_RANDOM_SEED")
    if random_seed is not None:
        _set_global_seed(random_seed)
        print(f"[MIGRA_CONFIG] random_seed={random_seed}")

    adata = sc.read_h5ad(str(input_h5ad))
    adata.obs_names_make_unique()
    ad.settings.allow_write_nullable_strings = True

    modality_key = _get_env_str("MRDR_MODALITY_KEY", "modality")
    batch_key = _get_env_str("MRDR_BATCH_KEY", "batch")
    input_layer = _resolve_input_layer(adata, os.environ.get("MRDR_INPUT_LAYER"))

    _require_obs_column(adata, modality_key)
    _require_obs_column(adata, batch_key)

    ignored_label_key = _infer_existing_label_key(adata)
    print(f"[MIGRA_CONFIG] modality_key={modality_key}, batch_key={batch_key}")
    print(f"[MIGRA_CONFIG] input_layer={input_layer if input_layer is not None else 'X'}")
    if ignored_label_key is not None:
        print(
            f"[MIGRA_CONFIG] detected label key {ignored_label_key!r}, "
            "but scMIGRA training keeps it out of the model to remain label-free."
        )
    else:
        print("[MIGRA_CONFIG] no label column detected; training remains fully label-free.")

    hidden_layers = _get_env_int_list("MRDR_HIDDEN_LAYERS", [256, 128])
    latent_dim_shared = _get_env_int("MRDR_LATENT_SHARED", 20)
    latent_dim_specific = _get_env_int("MRDR_LATENT_SPECIFIC", 16)
    latent_dim_tech = _get_env_int("MRDR_LATENT_TECH", 8)
    num_programs = _get_env_int("MRDR_NUM_PROGRAMS", 32)
    dropout_rate = _get_env_float("MRDR_DROPOUT_RATE", 0.2)

    beta_mod = _get_env_float("MRDR_BETA_MOD", 1.0)
    beta_tech = _get_env_float("MRDR_BETA_TECH", 1.0)
    lambda_program = _get_env_float("MRDR_LAMBDA_PROGRAM", 1.0)
    lambda_transport = _get_env_float("MRDR_LAMBDA_TRANSPORT", 1.0)
    lambda_cycle = _get_env_float("MRDR_LAMBDA_CYCLE", 0.5)
    lambda_batch_ind = _get_env_float("MRDR_LAMBDA_BATCH_IND", 0.1)

    program_sparsity_weight = _get_env_float("MRDR_PROGRAM_SPARSITY_WEIGHT", 1.0)
    program_orthogonality_weight = _get_env_float("MRDR_PROGRAM_ORTHOGONALITY_WEIGHT", 1.0)
    program_balance_weight = _get_env_float("MRDR_PROGRAM_BALANCE_WEIGHT", 0.10)

    anchor_topk = _get_env_int("MRDR_ANCHOR_TOPK", 2)
    local_k = _get_env_int("MRDR_LOCAL_K", 6)
    anchor_score_quantile = _get_env_float("MRDR_ANCHOR_SCORE_QUANTILE", 0.75)
    max_anchors = _get_env_int("MRDR_MAX_ANCHORS", 24)
    transport_temperature = _get_env_float("MRDR_TRANSPORT_TEMPERATURE", 0.20)
    transport_profile_weight = _get_env_float("MRDR_TRANSPORT_PROFILE_WEIGHT", 0.50)
    transport_graph_weight = _get_env_float("MRDR_TRANSPORT_GRAPH_WEIGHT", 0.50)
    cross_batch_bonus = _get_env_float("MRDR_CROSS_BATCH_BONUS", 0.05)

    epoch_num = _get_env_int("MRDR_EPOCH_NUM", 100)
    batch_size = _get_env_int("MRDR_BATCH_SIZE", 128)
    lr = _get_env_float("MRDR_LR", 1e-3)
    adaptlr = _get_env_bool("MRDR_ADAPTLR", True)
    warmup_epochs = _get_env_int("MRDR_WARMUP_EPOCHS", 10)
    early_stopping = _get_env_bool("MRDR_EARLY_STOPPING", False)
    valid_prop = _get_env_float("MRDR_VALID_PROP", 0.0)
    distribution = _get_env_str("MRDR_DISTRIBUTION", "ZINB")

    print("[MIGRA_CONFIG] hidden_layers=", hidden_layers)
    print(
        "[MIGRA_CONFIG] latent_shared={}, latent_specific={}, latent_tech={}, num_programs={}, dropout={}".format(
            latent_dim_shared,
            latent_dim_specific,
            latent_dim_tech,
            num_programs,
            dropout_rate,
        )
    )
    print(
        "[MIGRA_CONFIG] beta_mod={}, beta_tech={}, lambda_program={}, lambda_transport={}, lambda_cycle={}, lambda_batch_ind={}".format(
            beta_mod,
            beta_tech,
            lambda_program,
            lambda_transport,
            lambda_cycle,
            lambda_batch_ind,
        )
    )
    print(
        "[MIGRA_CONFIG] anchor_topk={}, local_k={}, score_quantile={}, max_anchors={}, temp={}, profile_w={}, graph_w={}, cross_batch_bonus={}".format(
            anchor_topk,
            local_k,
            anchor_score_quantile,
            max_anchors,
            transport_temperature,
            transport_profile_weight,
            transport_graph_weight,
            cross_batch_bonus,
        )
    )
    print(
        "[MIGRA_CONFIG] epoch_num={}, batch_size={}, lr={}, adaptlr={}, warmup_epochs={}, early_stopping={}, valid_prop={}, distribution={}".format(
            epoch_num,
            batch_size,
            lr,
            adaptlr,
            warmup_epochs,
            early_stopping,
            valid_prop,
            distribution,
        )
    )

    # 中文实验流程：
    # 1. 优先读取 counts 类图层；如果数据里没有对应 layer，就退回 adata.X。
    # 2. 训练阶段不把 cell type 标签送进模型，标签只留给后续评估使用。
    # 3. 训练结束后把 shared / specific / tech / programs 全部写回 h5ad，方便直接做指标。
    model = Integration(
        data=adata,
        layer=input_layer,
        modality_key=modality_key,
        batch_key=batch_key,
        celltype_key=None,
        feature_list=None,
        distribution=distribution,
    )

    model.setup(
        hidden_layers=hidden_layers,
        latent_dim_shared=latent_dim_shared,
        latent_dim_specific=latent_dim_specific,
        latent_dim_tech=latent_dim_tech,
        num_programs=num_programs,
        dropout_rate=dropout_rate,
        beta_mod=beta_mod,
        beta_tech=beta_tech,
        lambda_program=lambda_program,
        lambda_transport=lambda_transport,
        lambda_cycle=lambda_cycle,
        lambda_batch_ind=lambda_batch_ind,
        program_sparsity_weight=program_sparsity_weight,
        program_orthogonality_weight=program_orthogonality_weight,
        program_balance_weight=program_balance_weight,
        anchor_topk=anchor_topk,
        local_k=local_k,
        anchor_score_quantile=anchor_score_quantile,
        max_anchors=max_anchors,
        transport_temperature=transport_temperature,
        transport_profile_weight=transport_profile_weight,
        transport_graph_weight=transport_graph_weight,
        cross_batch_bonus=cross_batch_bonus,
    )

    model.train(
        epoch_num=epoch_num,
        batch_size=batch_size,
        lr=lr,
        adaptlr=adaptlr,
        warmup_epochs=warmup_epochs,
        early_stopping=early_stopping,
        valid_prop=valid_prop,
    )

    model.inference(n_samples=1, update=True, returns=False)
    trained = model.get_adata()
    trained.uns["migra_run_config"] = {
        "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "input_h5ad": str(input_h5ad),
        "output_h5ad": str(output_h5ad),
        "selected_layer": input_layer if input_layer is not None else "X",
        "modality_key": modality_key,
        "batch_key": batch_key,
        "ignored_label_key_for_training": ignored_label_key,
        "labels_used_for_supervision": False,
        "hidden_layers": hidden_layers,
        "latent_dim_shared": latent_dim_shared,
        "latent_dim_specific": latent_dim_specific,
        "latent_dim_tech": latent_dim_tech,
        "num_programs": num_programs,
        "dropout_rate": dropout_rate,
        "beta_mod": beta_mod,
        "beta_tech": beta_tech,
        "lambda_program": lambda_program,
        "lambda_transport": lambda_transport,
        "lambda_cycle": lambda_cycle,
        "lambda_batch_ind": lambda_batch_ind,
        "program_sparsity_weight": program_sparsity_weight,
        "program_orthogonality_weight": program_orthogonality_weight,
        "program_balance_weight": program_balance_weight,
        "anchor_topk": anchor_topk,
        "local_k": local_k,
        "anchor_score_quantile": anchor_score_quantile,
        "max_anchors": max_anchors,
        "transport_temperature": transport_temperature,
        "transport_profile_weight": transport_profile_weight,
        "transport_graph_weight": transport_graph_weight,
        "cross_batch_bonus": cross_batch_bonus,
        "epoch_num": epoch_num,
        "batch_size": batch_size,
        "lr": lr,
        "adaptlr": adaptlr,
        "warmup_epochs": warmup_epochs,
        "early_stopping": early_stopping,
        "valid_prop": valid_prop,
        "distribution": distribution,
        "output_embeddings": [
            "latent_shared",
            "latent_specific",
            "latent_tech",
            "program_assignments",
        ],
    }
    trained.write(str(output_h5ad))


if __name__ == "__main__":
    main()
