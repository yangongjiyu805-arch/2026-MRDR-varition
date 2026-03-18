import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBLoss(nn.Module):
    """Zero-inflated negative binomial loss for count reconstruction."""

    def __init__(self):
        super().__init__()

    def forward(self, x, rho, dispersion, pi, s, mask=None, eps=1e-8):
        mean = torch.clamp(rho * s, min=eps)
        dispersion = torch.clamp(dispersion, min=eps)

        t1 = torch.lgamma(dispersion) + torch.lgamma(x + 1.0) - torch.lgamma(x + dispersion)
        t2 = -dispersion * torch.log(dispersion) - x * torch.log(mean) + (dispersion + x) * torch.log(dispersion + mean)
        nb_final = t1 + t2

        zero_nb = torch.exp(dispersion * (torch.log(dispersion) - torch.log(dispersion + mean)))
        zero_case = -torch.log(pi + (1.0 - pi) * zero_nb + eps)
        nb_case = nb_final - torch.log(1.0 - pi + eps)

        loss = torch.where(x <= eps, zero_case, nb_case)
        if mask is not None:
            norm = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            mean_loss = torch.mean(torch.sum(loss * mask, dim=1) * (x.shape[1] / norm), dim=0)
        else:
            mean_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
        return mean_loss


def mse_loss(x, y, mask=None):
    loss = (x - y).pow(2)
    if mask is not None:
        norm = torch.clamp(torch.sum(mask, dim=1), min=1.0)
        mean_loss = torch.mean(torch.sum(loss * mask, dim=1) * (x.shape[1] / norm), dim=0)
    else:
        mean_loss = torch.mean(torch.sum(loss, dim=1), dim=0)
    return mean_loss


def kl_loss_prior(mu_q, logvar_q, mu_p, logvar_p):
    kl = -0.5 * torch.sum(
        -torch.exp(logvar_q - logvar_p)
        - ((mu_q - mu_p).pow(2)) * torch.exp(-logvar_p)
        + 1
        - logvar_p
        + logvar_q,
        dim=1,
    )
    return torch.mean(kl, dim=0)


def one_hot_to_label(y):
    if y is None or y.ndim != 2 or y.shape[1] == 0:
        return None
    if y.shape[1] == 1:
        return torch.zeros(y.shape[0], dtype=torch.long, device=y.device)
    return torch.argmax(y, dim=1)


def program_regularization(
    assignments,
    program_basis,
    sparsity_weight=1.0,
    orthogonality_weight=1.0,
    balance_weight=0.1,
    eps=1e-8,
):
    """program mixture 正则项。"""

    program_num = max(1, assignments.shape[1])
    entropy = -torch.sum(assignments * torch.log(assignments + eps), dim=1)
    sparsity_loss = torch.mean(entropy / math.log(program_num + eps))

    basis = F.normalize(program_basis, p=2, dim=1)
    gram = basis @ basis.t()
    identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    orthogonality_loss = torch.mean((gram - identity).pow(2))

    usage = assignments.mean(dim=0)
    uniform = torch.full_like(usage, 1.0 / max(1, usage.numel()))
    balance_loss = torch.sum(usage * (torch.log(usage + eps) - torch.log(uniform + eps)))

    total = (
        float(sparsity_weight) * sparsity_loss
        + float(orthogonality_weight) * orthogonality_loss
        + float(balance_weight) * balance_loss
    )
    stats = {
        "program_sparsity_loss": sparsity_loss,
        "program_orthogonality_loss": orthogonality_loss,
        "program_balance_loss": balance_loss,
    }
    return total, stats


def hsic_independence_loss(x, y):
    """线性 HSIC，用于约束 shared biology 与 batch 低相关。"""

    if y is None or y.ndim != 2 or y.shape[1] <= 1 or x.shape[0] <= 2:
        return torch.zeros((), device=x.device, dtype=x.dtype)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)
    kernel_x = x @ x.t()
    kernel_y = y @ y.t()

    n = x.shape[0]
    eye = torch.eye(n, device=x.device, dtype=x.dtype)
    centering = eye - (1.0 / n) * torch.ones((n, n), device=x.device, dtype=x.dtype)
    kernel_x = centering @ kernel_x @ centering
    kernel_y = centering @ kernel_y @ centering
    return torch.trace(kernel_x @ kernel_y) / max(1, (n - 1) ** 2)


def _build_local_summaries(features, modality_labels, k=6):
    features = F.normalize(features, p=2, dim=1)
    sim = features @ features.t()
    n = features.shape[0]
    neighbor_lists = []
    summaries = []

    for idx in range(n):
        same_modality = modality_labels == modality_labels[idx]
        same_modality[idx] = False
        candidates = torch.nonzero(same_modality, as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            neighbors = torch.tensor([idx], device=features.device, dtype=torch.long)
        else:
            k_eff = min(int(k), candidates.numel())
            top_idx = torch.topk(sim[idx, candidates], k=k_eff, largest=True).indices
            neighbors = torch.cat(
                [torch.tensor([idx], device=features.device, dtype=torch.long), candidates[top_idx]],
                dim=0,
            )
        neighbor_lists.append(neighbors)
        summaries.append(torch.mean(features[neighbors], dim=0))

    summaries = torch.stack(summaries, dim=0)
    return neighbor_lists, F.normalize(summaries, p=2, dim=1)


def _mutual_topk_pairs(score_matrix, cross_mask, anchor_topk=2, score_quantile=0.75, max_anchors=24):
    n = score_matrix.shape[0]
    candidate_mask = torch.zeros((n, n), device=score_matrix.device, dtype=torch.bool)

    for idx in range(n):
        candidates = torch.nonzero(cross_mask[idx], as_tuple=False).squeeze(1)
        if candidates.numel() == 0:
            continue
        k_eff = min(int(anchor_topk), candidates.numel())
        best = candidates[torch.topk(score_matrix[idx, candidates], k=k_eff, largest=True).indices]
        candidate_mask[idx, best] = True

    mutual = candidate_mask & candidate_mask.t()
    pair_idx = torch.nonzero(torch.triu(mutual, diagonal=1), as_tuple=False)
    if pair_idx.numel() == 0:
        return pair_idx, torch.zeros((0,), device=score_matrix.device, dtype=score_matrix.dtype)

    pair_scores = score_matrix[pair_idx[:, 0], pair_idx[:, 1]]
    threshold = torch.quantile(pair_scores.detach(), q=float(score_quantile)) if pair_scores.numel() > 1 else pair_scores[0]
    keep = pair_scores >= threshold
    if torch.any(keep):
        pair_idx = pair_idx[keep]
        pair_scores = pair_scores[keep]

    if pair_scores.numel() > int(max_anchors):
        top = torch.topk(pair_scores, k=int(max_anchors), largest=True).indices
        pair_idx = pair_idx[top]
        pair_scores = pair_scores[top]
    return pair_idx, pair_scores


def reliability_transport_loss(
    z_bio,
    assignments,
    common_profiles,
    modality_labels,
    batch_labels=None,
    anchor_topk=2,
    local_k=6,
    score_quantile=0.75,
    max_anchors=24,
    bio_weight=0.45,
    profile_weight=0.35,
    graph_weight=0.20,
    uncertainty_weight=0.15,
    cross_batch_bonus=0.05,
    coupling_temperature=0.20,
    transport_profile_weight=0.50,
    transport_graph_weight=0.50,
    eps=1e-8,
):
    """可靠锚点 + 局部运输损失。"""

    if z_bio.shape[0] <= 2:
        zero = torch.zeros((), device=z_bio.device, dtype=z_bio.dtype)
        return zero, {
            "transport_core_loss": zero,
            "cycle_loss": zero,
            "transport_profile_loss": zero,
            "transport_graph_loss": zero,
            "anchor_score_mean": zero,
            "anchor_coverage": zero,
            "anchor_count": zero,
        }

    if modality_labels is None or torch.unique(modality_labels).numel() <= 1:
        zero = torch.zeros((), device=z_bio.device, dtype=z_bio.dtype)
        return zero, {
            "transport_core_loss": zero,
            "cycle_loss": zero,
            "transport_profile_loss": zero,
            "transport_graph_loss": zero,
            "anchor_score_mean": zero,
            "anchor_coverage": zero,
            "anchor_count": zero,
        }

    z_norm = F.normalize(z_bio, p=2, dim=1)
    q_norm = F.normalize(assignments, p=2, dim=1)
    p_norm = F.normalize(common_profiles, p=2, dim=1)
    _, summary_norm = _build_local_summaries(q_norm, modality_labels, k=local_k)
    neighbor_lists, _ = _build_local_summaries(z_norm, modality_labels, k=local_k)

    bio_sim = z_norm @ z_norm.t()
    profile_sim = p_norm @ p_norm.t()
    summary_sim = summary_norm @ summary_norm.t()

    uncertainty = -torch.sum(assignments * torch.log(assignments + eps), dim=1)
    uncertainty = uncertainty / math.log(max(2, assignments.shape[1]))
    uncertainty_penalty = 0.5 * (uncertainty.unsqueeze(1) + uncertainty.unsqueeze(0))

    score = (
        float(bio_weight) * bio_sim
        + float(profile_weight) * profile_sim
        + float(graph_weight) * summary_sim
        - float(uncertainty_weight) * uncertainty_penalty
    )
    cross_modality_mask = modality_labels.unsqueeze(1) != modality_labels.unsqueeze(0)
    if batch_labels is not None and torch.unique(batch_labels).numel() > 1:
        score = score + float(cross_batch_bonus) * (batch_labels.unsqueeze(1) != batch_labels.unsqueeze(0)).float()
    score = score.masked_fill(~cross_modality_mask, -1e9)

    pair_idx, pair_scores = _mutual_topk_pairs(
        score,
        cross_modality_mask,
        anchor_topk=anchor_topk,
        score_quantile=score_quantile,
        max_anchors=max_anchors,
    )
    if pair_idx.numel() == 0:
        zero = torch.zeros((), device=z_bio.device, dtype=z_bio.dtype)
        return zero, {
            "transport_core_loss": zero,
            "cycle_loss": zero,
            "transport_profile_loss": zero,
            "transport_graph_loss": zero,
            "anchor_score_mean": zero,
            "anchor_coverage": zero,
            "anchor_count": zero,
        }

    transport_terms = []
    profile_terms = []
    graph_terms = []
    cycle_terms = []
    used_cells = set()

    for pair_no, (src_idx, tgt_idx) in enumerate(pair_idx.tolist()):
        used_cells.add(src_idx)
        used_cells.add(tgt_idx)
        src_neighbors = neighbor_lists[src_idx][: max(1, int(local_k))]
        tgt_neighbors = neighbor_lists[tgt_idx][: max(1, int(local_k))]

        src_z = z_norm[src_neighbors]
        tgt_z = z_norm[tgt_neighbors]
        src_q = q_norm[src_neighbors]
        tgt_q = q_norm[tgt_neighbors]
        src_p = p_norm[src_neighbors]
        tgt_p = p_norm[tgt_neighbors]

        cost = (
            0.50 * (1.0 - src_z @ tgt_z.t())
            + 0.30 * (1.0 - src_q @ tgt_q.t())
            + 0.20 * (1.0 - src_p @ tgt_p.t())
        )
        temperature = max(float(coupling_temperature), eps)
        transport_st = torch.softmax(-cost / temperature, dim=1)
        transport_ts = torch.softmax(-cost.t() / temperature, dim=1)

        bary_tgt_z = transport_st @ z_bio[tgt_neighbors]
        bary_tgt_p = transport_st @ common_profiles[tgt_neighbors]

        align_z = torch.mean((z_bio[src_neighbors] - bary_tgt_z).pow(2))
        align_p = torch.mean((common_profiles[src_neighbors] - bary_tgt_p).pow(2))

        dist_src = torch.cdist(z_norm[src_neighbors], z_norm[src_neighbors], p=2)
        dist_tgt = torch.cdist(z_norm[tgt_neighbors], z_norm[tgt_neighbors], p=2)
        transported_dist_tgt = transport_st @ dist_tgt @ transport_st.t()
        graph_reg = torch.mean((dist_src - transported_dist_tgt).pow(2))

        cycle_src = transport_st @ (transport_ts @ z_bio[src_neighbors])
        cycle_tgt = transport_ts @ (transport_st @ z_bio[tgt_neighbors])
        cycle_reg = torch.mean((z_bio[src_neighbors] - cycle_src).pow(2)) + torch.mean((z_bio[tgt_neighbors] - cycle_tgt).pow(2))

        anchor_weight = torch.sigmoid(pair_scores[pair_no])
        core_loss = align_z + float(transport_profile_weight) * align_p + float(transport_graph_weight) * graph_reg
        transport_terms.append(anchor_weight * core_loss)
        profile_terms.append(anchor_weight * align_p)
        graph_terms.append(anchor_weight * graph_reg)
        cycle_terms.append(anchor_weight * cycle_reg)

    transport_core = torch.stack(transport_terms).mean()
    profile_core = torch.stack(profile_terms).mean()
    graph_core = torch.stack(graph_terms).mean()
    cycle_core = torch.stack(cycle_terms).mean()

    anchor_score_mean = torch.mean(pair_scores)
    anchor_coverage = z_bio.new_tensor(len(used_cells) / max(1, z_bio.shape[0]))
    anchor_count = z_bio.new_tensor(float(pair_idx.shape[0]))
    stats = {
        "transport_core_loss": transport_core,
        "cycle_loss": cycle_core,
        "transport_profile_loss": profile_core,
        "transport_graph_loss": graph_core,
        "anchor_score_mean": anchor_score_mean,
        "anchor_coverage": anchor_coverage,
        "anchor_count": anchor_count,
    }
    return transport_core, stats
