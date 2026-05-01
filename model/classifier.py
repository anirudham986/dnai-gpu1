# =====================================================================
# model/classifier.py — NTv2 Dual-Sequence Variant Effect Classifier
#
# Architecture:
#   backbone (NT v2 100M) → mean_pool(REF) ⊕ mean_pool(ALT) ⊕ diff
#   → LayerNorm → Dropout → Linear(1536→256) → GELU → Dropout → Linear(256→2)
#
# The dual-sequence approach feeds both reference and alternate allele
# genomic contexts through the same backbone, then classifies based on
# the concatenation of both embeddings plus their difference. This
# captures both absolute context and the delta caused by the variant.
# =====================================================================

import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoConfig


class NTv2DualSeqClassifier(nn.Module):
    """
    Dual-sequence variant effect classifier based on NT v2.

    Input:  REF sequence tokens + ALT sequence tokens
    Output: 2-class logits (Benign=0, Pathogenic=1)

    Feature vector: [emb_ref ; emb_alt ; emb_ref - emb_alt]
    This 3-way concat gives the classifier access to:
      - Individual sequence representations
      - The explicit difference signal from the variant
    """

    def __init__(self, model_name: str, num_layers_to_unfreeze: int = 22,
                 dropout: float = 0.2):
        """
        Args:
            model_name: HuggingFace model name for NT v2
            num_layers_to_unfreeze: Number of transformer layers to fine-tune
                                     (22 = all layers = full fine-tuning)
            dropout: Dropout rate for the classification head
        """
        super().__init__()

        # Load backbone config
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        for attr, val in [('is_decoder', False),
                          ('add_cross_attention', False),
                          ('chunk_size_feed_forward', 0)]:
            if not hasattr(cfg, attr):
                setattr(cfg, attr, val)

        # Load pre-trained backbone (ESM architecture)
        full_model = AutoModelForMaskedLM.from_pretrained(
            model_name, config=cfg, trust_remote_code=True
        )
        self.backbone = full_model.esm
        del full_model  # Free the MLM head

        self.hidden_size = self.backbone.config.hidden_size    # 512
        n_layers = self.backbone.config.num_hidden_layers       # 22

        # ----- Freeze/Unfreeze Strategy -----
        freeze_until = n_layers - num_layers_to_unfreeze

        # Embeddings: only unfreeze for full fine-tuning
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = (num_layers_to_unfreeze >= n_layers)

        # Transformer layers: unfreeze top-K
        for i, layer in enumerate(self.backbone.encoder.layer):
            for p in layer.parameters():
                p.requires_grad = (i >= freeze_until)

        # Final layer norm: always trainable
        if hasattr(self.backbone, 'layer_norm'):
            for p in self.backbone.layer_norm.parameters():
                p.requires_grad = True

        # ----- Classification Head -----
        # Input: 3 * hidden_size (ref + alt + diff)
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size * 3),     # 1536
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size * 3, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

        # Xavier init for classification head
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def _embed(self, input_ids, attention_mask):
        """
        Mean pooling over backbone hidden states.
        This is the official NT v2 embedding approach.
        """
        out = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        return torch.sum(hidden * mask, dim=1) / torch.clamp(
            mask.sum(1), min=1e-9
        )

    def forward(self, ref_ids, ref_mask, alt_ids, alt_mask):
        """
        Forward pass with dual sequences.

        Args:
            ref_ids:  Reference sequence token IDs   [B, T]
            ref_mask: Reference attention mask         [B, T]
            alt_ids:  Alternate sequence token IDs     [B, T]
            alt_mask: Alternate attention mask          [B, T]

        Returns:
            Logits tensor of shape [B, 2]
        """
        emb_ref = self._embed(ref_ids, ref_mask)
        emb_alt = self._embed(alt_ids, alt_mask)
        features = torch.cat([emb_ref, emb_alt, emb_ref - emb_alt], dim=-1)
        return self.classifier(features)

    def get_param_groups(self, backbone_lr: float, head_lr: float) -> list:
        """
        Get parameter groups with differential learning rates.

        The backbone uses a very low LR to preserve pre-trained knowledge,
        while the classification head uses a higher LR to learn the task.
        """
        backbone_params = [p for p in self.backbone.parameters()
                           if p.requires_grad]
        head_params = list(self.classifier.parameters())
        return [
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': head_lr},
        ]
