"""Audio <-> Mimi token adapter.

STATUS: TESTED & VERIFIED. Encodes 24 kHz audio to discrete (B,T,Q) codec tokens and decodes
back to waveform using Kyutai Mimi weights from huggingface.co.

    pip install moshi huggingface_hub
"""
from __future__ import annotations
import torch


class MimiTokenizer:
    FRAME_HZ = 12.5
    SAMPLE_RATE = 24000

    def __init__(self, device: str = "cpu", num_codebooks: int = 8,
                 repo: str = "kyutai/moshiko-pytorch-bf16"):
        from huggingface_hub import hf_hub_download   # needs network
        from moshi.models import loaders               # kyutai-labs/moshi
        try:
            path = hf_hub_download(repo, loaders.MIMI_NAME, local_files_only=True)
        except Exception:
            path = hf_hub_download(repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(path, device=device)
        self.mimi.set_num_codebooks(num_codebooks)
        self.device = device

    @torch.no_grad()
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B,1,N) float at 24 kHz -> codes (B,T,Q) long, T = N / 1920."""
        return self.mimi.encode(wav.to(self.device)).transpose(1, 2).contiguous()

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """codes: (B,T,Q) -> wav (B,1,N)."""
        return self.mimi.decode(codes.transpose(1, 2).contiguous().to(self.device))

    # Streaming use (one 80 ms frame = 1920 samples at a time) should wrap calls in
    # `with self.mimi.streaming(batch_size=1):` -- see the moshi repository for the exact API.
