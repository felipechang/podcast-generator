import io

import numpy as np
import soundfile as sf


def concat_wav_bytes(parts: list[bytes]) -> tuple[bytes, int]:
    """Concatenate WAV blobs in order; all must share the same sample rate and channels."""
    if not parts:
        raise ValueError("No audio segments to concatenate")

    arrays: list[np.ndarray] = []
    sr: int | None = None
    channels: int | None = None

    for blob in parts:
        data, file_sr = sf.read(io.BytesIO(blob), dtype="float32")
        if sr is None:
            sr = int(file_sr)
            if data.ndim == 1:
                channels = 1
            else:
                channels = data.shape[1]
        elif int(file_sr) != sr:
            raise ValueError(f"Sample rate mismatch: expected {sr}, got {file_sr}")

        if data.ndim == 1:
            if channels != 1:
                raise ValueError("Channel layout mismatch across segments")
        else:
            if data.shape[1] != channels:
                raise ValueError("Channel layout mismatch across segments")

        arrays.append(data)

    merged = np.concatenate(arrays, axis=0)
    out = io.BytesIO()
    sf.write(out, merged, sr, format="WAV", subtype="PCM_16")
    return out.getvalue(), sr
