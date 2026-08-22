
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from genrec.modules.t5 import T5ForConditionalGeneration, T5Config
from typing import Any, Dict, Optional, Tuple


class Tiger(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.sem_id_dim = config.get("sem_id_dim", 3)
        self.embeddings_per_hierarchy = config.get(
            "embeddings_per_hierarchy",
            config.get("codebook_size", 256),
        )
        vocab_size = config.get(
            "vocab_size",
            self.embeddings_per_hierarchy * self.sem_id_dim + 1,
        )

        t5config = T5Config(
            num_layers=config.get("num_layers", 4),
            num_decoder_layers=config.get("num_decoder_layers", 4),
            d_model=config.get("d_model", 128),
            d_ff=config.get("d_ff", 1024),
            num_heads=config.get("num_heads", 6),
            d_kv=config.get("d_kv", 64),
            dropout_rate=config.get("dropout_rate", 0.1),
            vocab_size=vocab_size,
            pad_token_id=config.get("pad_token_id", 0),
            eos_token_id=config.get("eos_token_id", 0),
            decoder_start_token_id=config.get("pad_token_id", 0),
            feed_forward_proj=config.get("feed_forward_proj", "relu"),
        )
        self.model = T5ForConditionalGeneration(t5config)
        self.bias_debiaser = None
        # Kept for genrec PE RNG parity: Linear init + extra encoder dropout
        # even when the aux MSE weight is 0.
        self.pop_evidence_head = None

    @property
    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb = sum(
            p.numel()
            for p in self.model.get_input_embeddings().parameters()
            if p.requires_grad
        )
        return total, emb

    @staticmethod
    def hierarchy_logits_from_full(
        logits: torch.Tensor,
        sem_id_dim: int,
        embeddings_per_hierarchy: int,
    ) -> torch.Tensor:
        """Slice full-vocab logits [B, L, V] into per-level logits [B, L, E]."""
        levels = []
        for h in range(sem_id_dim):
            start = h * embeddings_per_hierarchy + 1
            end = start + embeddings_per_hierarchy
            levels.append(logits[:, h, start:end])
        return torch.stack(levels, dim=1)

    def _level_vocab_slice(self, level: int) -> Tuple[int, int]:
        e = self.embeddings_per_hierarchy
        start = level * e + 1
        return start, start + e

    def _encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ):
        return self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    def _maybe_debias_log_probs(
        self,
        log_probs: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        if self.bias_debiaser is None:
            return log_probs
        return self.bias_debiaser.debias_log_probs(log_probs, level)

    def _level_log_probs(
        self,
        encoder_outputs,
        attention_mask: Optional[torch.Tensor],
        decoder_input_ids: torch.Tensor,
        level: int,
    ) -> torch.Tensor:
        """Log-probabilities over codebook candidates at one hierarchy level."""
        decoder_out = self.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
        )
        sequence_output = decoder_out[0]
        if self.model.config.scale_decoder_outputs:
            sequence_output = sequence_output * (self.model.model_dim ** -0.5)

        hidden = sequence_output[:, -1, :]
        logits = self.model.lm_head(hidden)
        start, end = self._level_vocab_slice(level)
        log_probs = F.log_softmax(logits[:, start:end], dim=-1)
        return self._maybe_debias_log_probs(log_probs, level)

    def _encoder_last_hidden(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        enc_hidden = encoder_outputs[0]
        if attention_mask is not None:
            last_idx = attention_mask.long().sum(dim=1) - 1
        else:
            last_idx = torch.full(
                (enc_hidden.size(0),),
                enc_hidden.size(1) - 1,
                device=enc_hidden.device,
                dtype=torch.long,
            )
        batch_idx = torch.arange(enc_hidden.size(0), device=enc_hidden.device)
        return enc_hidden[batch_idx, last_idx]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        pop_rho_hat = None
        if self.pop_evidence_head is not None and labels is not None:
            hidden_sel = self._encoder_last_hidden(input_ids, attention_mask)
            pop_rho_hat = self.pop_evidence_head(hidden_sel.detach())

        return outputs.loss, outputs.logits, pop_rho_hat

    @torch.no_grad()
    def _hierarchical_beam_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        num_beams: int,
    ) -> torch.Tensor:
        device = input_ids.device
        batch_size = input_ids.size(0)
        num_beams = max(1, num_beams)
        e = self.embeddings_per_hierarchy
        h_dim = self.sem_id_dim
        pad_id = self.model.config.decoder_start_token_id

        encoder_outputs = self._encode(input_ids, attention_mask)
        enc_hidden = encoder_outputs[0]

        beam_scores = torch.zeros(batch_size, num_beams, device=device)
        if num_beams > 1:
            beam_scores[:, 1:] = -1e9
        offset_tokens = torch.zeros(
            batch_size, num_beams, h_dim, dtype=torch.long, device=device
        )

        for h in range(h_dim):
            if h == 0:
                dec_in = torch.full(
                    (batch_size, 1), pad_id, dtype=torch.long, device=device
                )
                enc_h = encoder_outputs
                enc_mask = attention_mask
            else:
                start_col = torch.full(
                    (batch_size, num_beams, 1), pad_id, dtype=torch.long, device=device
                )
                dec_in = torch.cat([start_col, offset_tokens[:, :, :h]], dim=-1)
                dec_in = dec_in.reshape(batch_size * num_beams, h + 1)
                enc_h = (enc_hidden.unsqueeze(1).expand(
                    -1, num_beams, -1, -1
                ).reshape(batch_size * num_beams, enc_hidden.size(1), enc_hidden.size(2)),)
                enc_mask = None
                if attention_mask is not None:
                    enc_mask = attention_mask.unsqueeze(1).expand(
                        -1, num_beams, -1
                    ).reshape(batch_size * num_beams, -1)

            log_probs = self._level_log_probs(enc_h, enc_mask, dec_in, h)

            if h == 0:
                top_scores, top_codes = torch.topk(log_probs, num_beams, dim=-1)
                beam_scores = top_scores
                offset_tokens[:, :, 0] = top_codes + 1
            else:
                log_probs = log_probs.view(batch_size, num_beams, e)
                total = beam_scores.unsqueeze(-1) + log_probs
                total = total.view(batch_size, num_beams * e)
                top_scores, top_idx = torch.topk(total, num_beams, dim=-1)
                prev_beam = top_idx // e
                code_idx = top_idx % e
                prev_beam_exp = prev_beam.unsqueeze(-1).expand(-1, -1, h)
                offset_tokens[:, :, :h] = torch.gather(
                    offset_tokens[:, :, :h], 1, prev_beam_exp
                )
                offset_tokens[:, :, h] = code_idx + h * e + 1
                beam_scores = top_scores

        start_col = torch.full(
            (batch_size * num_beams, 1), pad_id, dtype=torch.long, device=device
        )
        body = offset_tokens.reshape(batch_size * num_beams, h_dim)
        return torch.cat([start_col, body], dim=1)

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        num_beams: int = 20,
        **kwargs,
    ) -> torch.Tensor:
        decode_mode = kwargs.get("decode_mode", "beam")
        if decode_mode != "beam":
            raise ValueError(f"decode_mode must be 'beam', got {decode_mode!r}")
        return self._hierarchical_beam_generate(
            input_ids,
            attention_mask,
            num_beams=num_beams,
        )
