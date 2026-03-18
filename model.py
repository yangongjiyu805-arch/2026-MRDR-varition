import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import (
    ZINBLoss,
    hsic_independence_loss,
    kl_loss_prior,
    mse_loss,
    one_hot_to_label,
    program_regularization,
    reliability_transport_loss,
)


def sparsemax(logits, dim=-1):
    """稀疏 simplex 激活。"""

    logits = logits - logits.max(dim=dim, keepdim=True)[0]
    sorted_logits, _ = torch.sort(logits, dim=dim, descending=True)
    cumsum = torch.cumsum(sorted_logits, dim=dim)

    range_values = torch.arange(1, sorted_logits.shape[dim] + 1, device=logits.device, dtype=logits.dtype)
    view_shape = [1] * logits.ndim
    view_shape[dim] = -1
    range_values = range_values.view(view_shape)

    support = 1 + range_values * sorted_logits > cumsum
    support_size = support.sum(dim=dim, keepdim=True).clamp(min=1)
    tau = (torch.gather(cumsum, dim, support_size - 1) - 1) / support_size
    return torch.clamp(logits - tau, min=0.0)


def _mlp(input_dim, layer_dims, dropout_rate=0.2):
    layers = []
    current_dim = input_dim
    for dim in layer_dims:
        layers.append(nn.Linear(current_dim, dim))
        layers.append(nn.BatchNorm1d(dim))
        layers.append(nn.LeakyReLU())
        layers.append(nn.Dropout(dropout_rate))
        current_dim = dim
    return nn.Sequential(*layers)


class ProgramStateEncoder(nn.Module):
    """Program-simplex biology encoder。

    中文总结:
    1. 输入公共特征空间表达与模态信息。
    2. 输出稀疏的 program mixture `a`。
    3. 通过可学习的 program basis 将 `a` 投影成共享 biology latent。
    """

    def __init__(self, input_dim, layer_dims, num_programs, latent_dim_bio, dropout_rate=0.2):
        super().__init__()
        self.backbone = _mlp(input_dim, layer_dims, dropout_rate=dropout_rate)
        hidden_dim = layer_dims[-1]
        self.logit_layer = nn.Linear(hidden_dim, num_programs)
        self.program_basis = nn.Parameter(torch.empty(num_programs, latent_dim_bio))
        nn.init.xavier_uniform_(self.program_basis)

    def forward(self, x):
        hidden = self.backbone(x)
        logits = self.logit_layer(hidden)
        assignments = sparsemax(logits, dim=-1)
        assignments = assignments / torch.clamp(assignments.sum(dim=1, keepdim=True), min=1e-8)
        program_basis = F.normalize(self.program_basis, p=2, dim=1)
        z_bio = assignments @ program_basis
        return assignments, z_bio, logits


class GaussianEncoder(nn.Module):
    def __init__(self, input_dim, layer_dims, latent_dim, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim <= 0:
            self.backbone = None
            self.mu_layer = None
            self.logvar_layer = None
        else:
            self.backbone = _mlp(input_dim, layer_dims, dropout_rate=dropout_rate)
            hidden_dim = layer_dims[-1]
            self.mu_layer = nn.Linear(hidden_dim, latent_dim)
            self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        if self.latent_dim <= 0:
            batch_size = x.shape[0]
            empty = x.new_zeros((batch_size, 0))
            return empty, empty, empty
        hidden = self.backbone(x)
        mu = self.mu_layer(hidden)
        logvar = self.logvar_layer(hidden)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return z, mu, logvar


class ConditionalPriorNet(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim=64, dropout_rate=0.2):
        super().__init__()
        self.latent_dim = latent_dim
        if latent_dim <= 0:
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, 2 * latent_dim),
            )

    def forward(self, x):
        if self.latent_dim <= 0:
            batch_size = x.shape[0]
            empty = x.new_zeros((batch_size, 0))
            return empty, empty
        params = self.net(x)
        mu, logvar = torch.chunk(params, 2, dim=-1)
        return mu, logvar


class MSEDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, layer_dims, dropout_rate=0.2, positive_outputs=True):
        super().__init__()
        hidden_layers = list(reversed(layer_dims))
        self.decoder = _mlp(latent_dim, hidden_layers, dropout_rate=dropout_rate)
        hidden_dim = hidden_layers[-1]
        if positive_outputs:
            self.mean_layer = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softplus())
        else:
            self.mean_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, z):
        hidden = self.decoder(z)
        return self.mean_layer(hidden)


class ZINBDecoder(nn.Module):
    def __init__(self, input_dim, latent_dim, modality_num, layer_dims, dropout_rate=0.2):
        super().__init__()
        hidden_layers = list(reversed(layer_dims))
        self.decoder = _mlp(latent_dim, hidden_layers, dropout_rate=dropout_rate)
        hidden_dim = hidden_layers[-1]
        self.mean_layer = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Softmax(dim=-1))
        self.dispersion_modality = nn.Parameter(torch.randn(modality_num, input_dim))
        self.dropout_layer = nn.Sequential(nn.Linear(hidden_dim, input_dim), nn.Sigmoid())

    def forward(self, z, modality_onehot):
        hidden = self.decoder(z)
        rho = self.mean_layer(hidden)
        dispersion = torch.exp(modality_onehot @ self.dispersion_modality)
        pi = self.dropout_layer(hidden)
        return rho, dispersion, pi


class NBDecoder(ZINBDecoder):
    def forward(self, z, modality_onehot):
        rho, dispersion, _ = super().forward(z, modality_onehot)
        pi = torch.zeros_like(rho)
        return rho, dispersion, pi


class EmbeddingNet(nn.Module):
    """scMIGRA-v1 主体。"""

    def __init__(
        self,
        device,
        input_dim,
        modality_num,
        covariate_dim=1,
        layer_dims=None,
        latent_dim_shared=20,
        latent_dim_specific=16,
        latent_dim_tech=8,
        num_programs=32,
        dropout_rate=0.2,
        beta_mod=1.0,
        beta_tech=1.0,
        lambda_program=1.0,
        lambda_transport=1.0,
        lambda_cycle=0.5,
        lambda_batch_ind=0.1,
        program_sparsity_weight=1.0,
        program_orthogonality_weight=1.0,
        program_balance_weight=0.10,
        anchor_topk=2,
        local_k=6,
        anchor_score_quantile=0.75,
        max_anchors=24,
        transport_temperature=0.20,
        transport_profile_weight=0.50,
        transport_graph_weight=0.50,
        cross_batch_bonus=0.05,
        feat_mask=None,
        distribution="ZINB",
        eps=1e-10,
    ):
        super().__init__()
        if layer_dims is None:
            layer_dims = [256, 128]

        self.device = device
        self.eps = eps
        self.input_dim = input_dim
        self.modality_num = modality_num
        self.covariate_dim = covariate_dim
        self.latent_dim_bio = latent_dim_shared
        self.latent_dim_mod = latent_dim_specific
        self.latent_dim_tech = latent_dim_tech
        self.num_programs = num_programs

        self.beta_mod = beta_mod
        self.beta_tech = beta_tech
        self.lambda_program = lambda_program
        self.lambda_transport = lambda_transport
        self.lambda_cycle = lambda_cycle
        self.lambda_batch_ind = lambda_batch_ind

        self.program_sparsity_weight = program_sparsity_weight
        self.program_orthogonality_weight = program_orthogonality_weight
        self.program_balance_weight = program_balance_weight

        self.anchor_topk = anchor_topk
        self.local_k = local_k
        self.anchor_score_quantile = anchor_score_quantile
        self.max_anchors = max_anchors
        self.transport_temperature = transport_temperature
        self.transport_profile_weight = transport_profile_weight
        self.transport_graph_weight = transport_graph_weight
        self.cross_batch_bonus = cross_batch_bonus

        self.feat_mask = feat_mask.to(self.device) if feat_mask is not None else None

        self.distribution = distribution
        if distribution in ["ZINB", "NB"]:
            self.count_data = True
            self.positive_outputs = True
        elif distribution == "Normal":
            self.count_data = False
            self.positive_outputs = False
        else:
            self.count_data = False
            self.positive_outputs = True

        state_input_dim = input_dim + modality_num
        self.state_encoder = ProgramStateEncoder(
            input_dim=state_input_dim,
            layer_dims=layer_dims,
            num_programs=num_programs,
            latent_dim_bio=latent_dim_shared,
            dropout_rate=dropout_rate,
        )
        self.mod_encoder = GaussianEncoder(
            input_dim=input_dim + modality_num,
            layer_dims=layer_dims,
            latent_dim=latent_dim_specific,
            dropout_rate=dropout_rate,
        )
        tech_input_dim = input_dim + (covariate_dim if covariate_dim > 0 else 0)
        self.tech_encoder = GaussianEncoder(
            input_dim=tech_input_dim,
            layer_dims=layer_dims,
            latent_dim=latent_dim_tech,
            dropout_rate=dropout_rate,
        )

        self.mod_prior = ConditionalPriorNet(
            input_dim=num_programs + modality_num,
            latent_dim=latent_dim_specific,
            hidden_dim=max(64, layer_dims[-1]),
            dropout_rate=dropout_rate,
        )
        self.tech_prior = ConditionalPriorNet(
            input_dim=max(covariate_dim, 1),
            latent_dim=latent_dim_tech,
            hidden_dim=max(32, layer_dims[-1] // 2),
            dropout_rate=dropout_rate,
        )

        decoder_input_dim = latent_dim_shared + latent_dim_specific + latent_dim_tech + modality_num
        if distribution == "ZINB":
            self.decoder = ZINBDecoder(
                input_dim=input_dim,
                latent_dim=decoder_input_dim,
                modality_num=modality_num,
                layer_dims=layer_dims,
                dropout_rate=dropout_rate,
            )
        elif distribution == "NB":
            self.decoder = NBDecoder(
                input_dim=input_dim,
                latent_dim=decoder_input_dim,
                modality_num=modality_num,
                layer_dims=layer_dims,
                dropout_rate=dropout_rate,
            )
        else:
            self.decoder = MSEDecoder(
                input_dim=input_dim,
                latent_dim=decoder_input_dim,
                layer_dims=layer_dims,
                dropout_rate=dropout_rate,
                positive_outputs=self.positive_outputs,
            )

    def _encode(self, x, b, m):
        model_x = torch.log1p(x) if self.count_data else x
        assignments, z_bio, _ = self.state_encoder(torch.cat([model_x, m], dim=-1))
        z_mod, mu_mod, logvar_mod = self.mod_encoder(torch.cat([model_x, m], dim=-1))

        if self.covariate_dim > 0:
            tech_context = torch.cat([model_x, b], dim=-1)
        else:
            tech_context = model_x
        z_tech, mu_tech, logvar_tech = self.tech_encoder(tech_context)
        return {
            "model_x": model_x,
            "assignments": assignments,
            "z_bio": z_bio,
            "z_mod": z_mod,
            "mu_mod": mu_mod,
            "logvar_mod": logvar_mod,
            "z_tech": z_tech,
            "mu_tech": mu_tech,
            "logvar_tech": logvar_tech,
        }

    def _decode(self, z_bio, z_mod, z_tech, modality_onehot):
        latent = torch.cat([z_bio, z_mod, z_tech, modality_onehot], dim=-1)
        if self.count_data:
            rho, dispersion, pi = self.decoder(latent, modality_onehot)
            common_profile = rho
        else:
            rho = self.decoder(latent)
            dispersion = None
            pi = None
            common_profile = torch.softmax(rho, dim=-1)
        return rho, dispersion, pi, common_profile

    def forward(self, x, b, m, i, w, stage="train"):
        del w, stage
        encoded = self._encode(x, b, m)
        rho, dispersion, pi, common_profile = self._decode(
            encoded["z_bio"],
            encoded["z_mod"],
            encoded["z_tech"],
            m,
        )

        if self.feat_mask is not None:
            mask = i @ self.feat_mask
        else:
            mask = None

        if self.count_data:
            sequencing_depth = self.sample_sequencing_depth(x)
            recon_loss = ZINBLoss()(x, rho, dispersion, pi, sequencing_depth, mask, eps=self.eps)
        else:
            recon_loss = mse_loss(x, rho, mask)

        prior_mod_mu, prior_mod_logvar = self.mod_prior(torch.cat([encoded["assignments"], m], dim=-1))
        kl_mod = kl_loss_prior(encoded["mu_mod"], encoded["logvar_mod"], prior_mod_mu, prior_mod_logvar)

        if self.latent_dim_tech > 0:
            if self.covariate_dim > 0 and b.ndim == 2 and b.shape[1] > 0:
                prior_batch_input = b
            else:
                prior_batch_input = x.new_zeros((x.shape[0], 1))
            prior_tech_mu, prior_tech_logvar = self.tech_prior(prior_batch_input)
            kl_tech = kl_loss_prior(encoded["mu_tech"], encoded["logvar_tech"], prior_tech_mu, prior_tech_logvar)
        else:
            kl_tech = torch.zeros((), device=x.device, dtype=x.dtype)

        program_loss, program_stats = program_regularization(
            encoded["assignments"],
            self.state_encoder.program_basis,
            sparsity_weight=self.program_sparsity_weight,
            orthogonality_weight=self.program_orthogonality_weight,
            balance_weight=self.program_balance_weight,
            eps=self.eps,
        )

        modality_labels = one_hot_to_label(m)
        batch_labels = one_hot_to_label(b) if b.ndim == 2 and b.shape[1] > 1 else None
        transport_core, transport_stats = reliability_transport_loss(
            encoded["z_bio"],
            encoded["assignments"],
            common_profile,
            modality_labels,
            batch_labels=batch_labels,
            anchor_topk=self.anchor_topk,
            local_k=self.local_k,
            score_quantile=self.anchor_score_quantile,
            max_anchors=self.max_anchors,
            coupling_temperature=self.transport_temperature,
            transport_profile_weight=self.transport_profile_weight,
            transport_graph_weight=self.transport_graph_weight,
            cross_batch_bonus=self.cross_batch_bonus,
            eps=self.eps,
        )

        batch_ind_loss = hsic_independence_loss(encoded["z_bio"], b if b.ndim == 2 and b.shape[1] > 1 else None)

        total_loss = (
            recon_loss
            + self.beta_mod * kl_mod
            + self.beta_tech * kl_tech
            + self.lambda_program * program_loss
            + self.lambda_transport * transport_core
            + self.lambda_cycle * transport_stats["cycle_loss"]
            + self.lambda_batch_ind * batch_ind_loss
        )

        loss_dict = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "kl_mod": kl_mod.item(),
            "kl_tech": kl_tech.item(),
            "program_loss": program_loss.item(),
            "transport_loss": transport_core.item(),
            "cycle_loss": transport_stats["cycle_loss"].item(),
            "batch_ind_loss": batch_ind_loss.item(),
            "program_sparsity_loss": program_stats["program_sparsity_loss"].item(),
            "program_orthogonality_loss": program_stats["program_orthogonality_loss"].item(),
            "program_balance_loss": program_stats["program_balance_loss"].item(),
            "transport_profile_loss": transport_stats["transport_profile_loss"].item(),
            "transport_graph_loss": transport_stats["transport_graph_loss"].item(),
            "anchor_score_mean": transport_stats["anchor_score_mean"].item(),
            "anchor_coverage": transport_stats["anchor_coverage"].item(),
            "anchor_count": transport_stats["anchor_count"].item(),
        }
        aux = {
            "z_tech": encoded["z_tech"],
            "assignments": encoded["assignments"],
            "common_profile": common_profile,
        }
        return encoded["z_bio"], encoded["z_mod"], total_loss, loss_dict, aux

    def decode_from_latent(self, z_bio, z_mod, modality_onehot, z_tech=None, library_size=None):
        if z_tech is None:
            z_tech = torch.zeros((z_bio.shape[0], self.latent_dim_tech), device=z_bio.device, dtype=z_bio.dtype)

        rho, _, _, _ = self._decode(z_bio, z_mod, z_tech, modality_onehot)
        if self.count_data:
            if library_size is None:
                scale = torch.ones((z_bio.shape[0], 1), device=z_bio.device, dtype=z_bio.dtype)
            else:
                scale = library_size.reshape(-1, 1).to(z_bio.device)
            return rho * scale
        return rho

    @staticmethod
    def sample_sequencing_depth(x, strategy="observed"):
        if strategy == "batch_sample":
            mu_s = torch.log(x.sum(dim=1) + 1.0).mean()
            sigma_s = torch.log(x.sum(dim=1) + 1.0).std()
            log_s = mu_s + sigma_s * torch.randn(1, device=x.device)
            s = torch.exp(log_s).repeat(x.shape[0]).unsqueeze(1)
        else:
            log_s = torch.log(torch.clamp(x.sum(dim=1), min=1e-8)).unsqueeze(1)
            s = torch.exp(log_s)
        return s
