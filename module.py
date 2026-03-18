import numpy as np
import torch
import torch.utils.data as Data
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import anndata as ad

from .data import CombinedDataset
from .model import EmbeddingNet
from .train import inference_model, train_model

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


def to_model_array(x):
    if issparse(x):
        return x.tocsr(copy=True)
    if isinstance(x, np.ndarray):
        return x.copy()
    raise TypeError(f"Unsupported type: {type(x)}")


class Integration:
    """scMIGRA-v1 外层接口。"""

    def __init__(
        self,
        data,
        layer=None,
        modality_key="modality",
        batch_key=None,
        celltype_key=None,
        distribution="ZINB",
        mask_key=None,
        feature_list=None,
        auto_make_obs_unique=True,
    ):
        super().__init__()

        if isinstance(data, list) and isinstance(data[0], ad.AnnData):
            self.adata = ad.concat(data, axis="obs", join="inner", label="modality")
        elif isinstance(data, ad.AnnData):
            self.adata = data
        else:
            raise ValueError("Wrong type of data!")

        if auto_make_obs_unique and (not self.adata.obs_names.is_unique):
            self.adata.obs_names_make_unique()

        if layer is None:
            self.data = to_model_array(self.adata.X)
        else:
            self.data = to_model_array(self.adata.layers[layer])

        label_encoder = LabelEncoder()
        onehot_encoder = OneHotEncoder(sparse_output=False)

        self.modality_label = self.adata.obs[modality_key].to_numpy()
        self.modality = label_encoder.fit_transform(self.modality_label)
        self.modality_ordered = [label for label in label_encoder.classes_]
        self.modality = onehot_encoder.fit_transform(self.modality.reshape(-1, 1))

        if celltype_key is None:
            self.celltype = None
            self.celltype_ordered = None
        else:
            self.celltype_label = self.adata.obs[celltype_key].to_numpy()
            self.celltype = label_encoder.fit_transform(self.celltype_label)
            self.celltype_ordered = [label for label in label_encoder.classes_]
            self.celltype = onehot_encoder.fit_transform(self.celltype.reshape(-1, 1))

        if batch_key is None:
            self.covariates = None
            self.covariates_ordered = None
        else:
            self.covariates_label = self.adata.obs[batch_key].to_numpy()
            self.covariates = label_encoder.fit_transform(self.covariates_label)
            self.covariates_ordered = [label for label in label_encoder.classes_]
            self.covariates = onehot_encoder.fit_transform(self.covariates.reshape(-1, 1))

        self.modality_num = self.modality.shape[1]
        self.celltype_num = self.celltype.shape[1] if self.celltype is not None else 0
        self.covariates_dim = self.covariates.shape[1] if self.covariates is not None else 0

        if mask_key is None:
            self.mask = self.modality
            self.mask_num = self.modality_num
            self.mask_ordered = self.modality_ordered
        else:
            self.mask_label = self.adata.obs[mask_key].to_numpy()
            self.mask = label_encoder.fit_transform(self.mask_label)
            self.mask_ordered = [label for label in label_encoder.classes_]
            self.mask = onehot_encoder.fit_transform(self.mask.reshape(-1, 1))
            self.mask_num = self.mask.shape[1]

        if feature_list is not None:
            self.feat_mask = torch.zeros(self.mask_num, self.data.shape[1])
            feature_list_ordered = [feature_list[label] for label in self.mask_ordered]
            for idx, feat_idx in enumerate(feature_list_ordered):
                self.feat_mask[idx, feat_idx] = 1
        else:
            self.feat_mask = torch.ones(self.mask_num, self.data.shape[1])

        self.distribution = distribution
        if distribution not in ["ZINB", "NB", "Normal", "Normal_positive"]:
            raise ValueError("Distribution not recognized!")

    def setup(
        self,
        hidden_layers=None,
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
        device=None,
    ):
        if hidden_layers is None:
            hidden_layers = [256, 128]

        self.input_dim = self.data.shape[1]
        self.hidden_layers = hidden_layers
        self.latent_dim_shared = latent_dim_shared
        self.latent_dim_specific = latent_dim_specific
        self.latent_dim_tech = latent_dim_tech
        self.num_programs = num_programs
        self.dropout_rate = dropout_rate

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

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = device
        print("using " + str(self.device))

        self.model = EmbeddingNet(
            device=self.device,
            input_dim=self.input_dim,
            modality_num=self.modality_num,
            covariate_dim=self.covariates_dim,
            layer_dims=self.hidden_layers,
            latent_dim_shared=self.latent_dim_shared,
            latent_dim_specific=self.latent_dim_specific,
            latent_dim_tech=self.latent_dim_tech,
            num_programs=self.num_programs,
            dropout_rate=self.dropout_rate,
            beta_mod=self.beta_mod,
            beta_tech=self.beta_tech,
            lambda_program=self.lambda_program,
            lambda_transport=self.lambda_transport,
            lambda_cycle=self.lambda_cycle,
            lambda_batch_ind=self.lambda_batch_ind,
            program_sparsity_weight=self.program_sparsity_weight,
            program_orthogonality_weight=self.program_orthogonality_weight,
            program_balance_weight=self.program_balance_weight,
            anchor_topk=self.anchor_topk,
            local_k=self.local_k,
            anchor_score_quantile=self.anchor_score_quantile,
            max_anchors=self.max_anchors,
            transport_temperature=self.transport_temperature,
            transport_profile_weight=self.transport_profile_weight,
            transport_graph_weight=self.transport_graph_weight,
            cross_batch_bonus=self.cross_batch_bonus,
            feat_mask=self.feat_mask,
            distribution=self.distribution,
        ).to(self.device)

        self.train_dataset = CombinedDataset(self.data, self.covariates, self.modality, self.mask, self.celltype)

    def train(
        self,
        epoch_num=200,
        batch_size=64,
        lr=1e-4,
        accumulation_steps=1,
        adaptlr=False,
        valid_prop=0.1,
        early_stopping=True,
        patience=10,
        weighted=False,
        tensorboard=False,
        savepath="./",
        random_state=42,
        warmup_epochs=10,
    ):
        if tensorboard and SummaryWriter is None:
            print("TensorBoard is not available in current environment; continue without it.")
            self.writer = None
        else:
            self.writer = SummaryWriter(savepath) if tensorboard else None

        self.epoch_num = epoch_num
        self.batch_size = batch_size
        self.lr = lr
        self.accumulation_steps = accumulation_steps
        self.adaptlr = adaptlr

        if valid_prop > 0:
            train_indices, valid_indices = train_test_split(
                np.arange(len(self.train_dataset)),
                test_size=valid_prop,
                stratify=self.modality.argmax(-1),
                random_state=random_state,
            )
            train_dataset = Data.Subset(self.train_dataset, train_indices)
            valid_dataset = Data.Subset(self.train_dataset, valid_indices)
        else:
            train_dataset, valid_dataset = self.train_dataset, self.train_dataset
            train_indices = np.arange(len(self.train_dataset))

        self.num_batch = max(1, len(train_dataset) // self.batch_size)

        print("Training start!")
        if weighted:
            weights = 1.0 / np.bincount(self.modality.argmax(-1))
            sample_weights = weights[self.modality.argmax(-1)]
            sample_weights = sample_weights[train_indices]
        else:
            sample_weights = None

        train_model(
            self.device,
            self.writer,
            train_dataset,
            valid_dataset,
            self.model,
            self.epoch_num,
            self.batch_size,
            self.num_batch,
            self.lr,
            accumulation_steps=self.accumulation_steps,
            adaptlr=self.adaptlr,
            early_stopping=early_stopping,
            patience=patience,
            sample_weights=sample_weights,
            warmup_epochs=warmup_epochs,
        )

        if self.writer is not None:
            self.writer.close()
        print("Training finished!")

    def inference(self, n_samples=1, dataset=None, batch_size=None, update=True, returns=False):
        if dataset is None:
            dataset = self.train_dataset
        if batch_size is None:
            batch_size = self.batch_size

        if n_samples > 1:
            z_bio, z_mod, z_tech, assignments = zip(
                *[inference_model(self.device, dataset, self.model, batch_size) for _ in range(n_samples)]
            )
            self.z_shared = np.mean(np.stack(z_bio, axis=0), axis=0)
            self.z_specific = np.mean(np.stack(z_mod, axis=0), axis=0)
            self.z_tech = np.mean(np.stack(z_tech, axis=0), axis=0)
            self.program_assignments = np.mean(np.stack(assignments, axis=0), axis=0)
        else:
            self.z_shared, self.z_specific, self.z_tech, self.program_assignments = inference_model(
                self.device,
                dataset,
                self.model,
                batch_size,
            )

        if update:
            self.adata.obsm["latent_shared"] = self.z_shared
            self.adata.obsm["latent_specific"] = self.z_specific
            self.adata.obsm["latent_tech"] = self.z_tech
            self.adata.obsm["program_assignments"] = self.program_assignments
            self.adata.uns["migra_program_basis"] = self.model.state_encoder.program_basis.detach().cpu().numpy()
            print("All results recorded in adata.")

        if returns:
            return self.z_shared, self.z_specific

    def generate_from_latent(self, z_bio, z_mod, modality, z_tech=None, library_size=None):
        z_bio_t = torch.tensor(z_bio, dtype=torch.float32, device=self.device)
        z_mod_t = torch.tensor(z_mod, dtype=torch.float32, device=self.device)
        m_t = torch.tensor(modality, dtype=torch.float32, device=self.device)

        if z_tech is not None:
            z_tech_t = torch.tensor(z_tech, dtype=torch.float32, device=self.device)
        else:
            z_tech_t = None

        if library_size is not None:
            l_t = torch.tensor(library_size, dtype=torch.float32, device=self.device)
        else:
            l_t = None

        self.model.eval()
        with torch.no_grad():
            x_pred = self.model.decode_from_latent(z_bio_t, z_mod_t, m_t, z_tech=z_tech_t, library_size=l_t)
        return x_pred.detach().cpu().numpy()

    def _program_mod_bank(self, q_target, z_mod_target, eps=1e-6):
        weights = q_target.sum(axis=0, keepdims=True).T
        bank = (q_target.T @ z_mod_target) / np.clip(weights, eps, None)
        global_mean = np.mean(z_mod_target, axis=0, keepdims=True)
        low_support = weights.squeeze(1) < eps
        if np.any(low_support):
            bank[low_support] = global_mean
        return bank

    def predict(self, predict_modality, batch_size=None, strategy="latent", method="program", k=10, library_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        del batch_size

        if strategy != "latent":
            raise ValueError("scMIGRA-v1 only supports strategy='latent'.")

        curr_index = self.modality_label == predict_modality
        impt_index = self.modality_label != predict_modality
        if not np.any(curr_index):
            raise ValueError(f"No cells found for predict_modality={predict_modality!r}")

        z_bio_curr = self.z_shared[curr_index, :]
        z_mod_curr = self.z_specific[curr_index, :]
        z_bio_impt = self.z_shared[impt_index, :]

        if method == "program":
            q_curr = self.program_assignments[curr_index, :]
            q_impt = self.program_assignments[impt_index, :]
            mod_bank = self._program_mod_bank(q_curr, z_mod_curr)
            z_mod_pred = q_impt @ mod_bank
        elif method == "knn":
            nbrs = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(z_bio_curr)
            distances, indices = nbrs.kneighbors(z_bio_impt)
            weights = 1.0 / (distances + 1e-5)
            weights = weights / np.sum(weights, axis=1, keepdims=True)
            z_mod_pred = np.array([
                np.sum(z_mod_curr[indices[idx]] * weights[idx][:, np.newaxis], axis=0) for idx in range(indices.shape[0])
            ])
        else:
            raise ValueError("Unknown method! Use 'program' or 'knn'.")

        z_tech_pred = self.z_tech[impt_index, :] if self.z_tech.shape[1] > 0 else None
        modality = np.tile(self.modality[curr_index, :][0, :], (z_bio_impt.shape[0], 1))
        return self.generate_from_latent(
            z_bio_impt,
            z_mod_pred,
            modality,
            z_tech=z_tech_pred,
            library_size=library_size,
        )

    def get_adata(self):
        return self.adata
