import numpy as np
from typing import Tuple

class ScalarQuantizerSQ8:
    """
    Asymmetric 8-bit Scalar Quantizer (SQ8) mapping float32 vectors to uint8 [0, 255].
    Achieves a 4.0x reduction in vector memory footprint with minimal recall degradation.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.vmin: np.ndarray | None = None
        self.vmax: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.is_calibrated: bool = False

    def calibrate(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected shape (N, {self.dim}), got {vectors.shape}")

        self.vmin = np.min(vectors, axis=0).astype(np.float32)
        self.vmax = np.max(vectors, axis=0).astype(np.float32)
        span = self.vmax - self.vmin
        self.scale = np.where(span < 1e-8, 1.0, span / 255.0).astype(np.float32)
        self.is_calibrated = True

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Compresses float32 array into uint8 representation."""
        single = vectors.ndim == 1
        if single:
            vectors = vectors[np.newaxis, :]

        if not self.is_calibrated:
            self.calibrate(vectors)

        scaled = np.round((vectors - self.vmin) / self.scale)
        quantized = np.clip(scaled, 0, 255).astype(np.uint8)
        return quantized[0] if single else quantized

    def decode(self, quantized: np.ndarray) -> np.ndarray:
        """Reconstructs approximate float32 vectors from uint8 codes."""
        if not self.is_calibrated:
            raise RuntimeError("Quantizer must be calibrated before decoding.")
        return (quantized.astype(np.float32) * self.scale) + self.vmin

    def compute_reconstruction_mse(self, original: np.ndarray) -> float:
        """Computes Mean Squared Quantization Error (MSQE) for academic benchmarking."""
        compressed = self.encode(original)
        reconstructed = self.decode(compressed)
        return float(np.mean((original - reconstructed) ** 2))
