from __future__ import annotations

import datetime as dt
import io
from pathlib import Path
from typing import Optional
import time

import numpy as np
import streamlit as st
import torch

from audio_utils import (
    SAMPLE_RATE,
    AudioProcessingResult,
    build_features,
    logmel_to_tensor,
    save_waveform,
)
from model_utils import LabelMaps, load_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_MAP_PATH = PROJECT_ROOT / "label_maps_google_tts.json"
MODEL_CHOICES: dict[str, Path] = {
    "Stage-2 Manual (Colab)": PROJECT_ROOT / "cnn_bilstm_stage2_manual-colab.pt",
    "Stage-2 Grid (lr1e-4_ep8_bs8)": PROJECT_ROOT / "cnn_bilstm_stage2_grid_lr0.0001_bs8_mw3_lr0.0001_ep8_bs8.pt",
}
UPLOAD_DIR = PROJECT_ROOT / "streamlit_uploads"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RECORDER_SAMPLE_RATE = 44_100

st.set_page_config(page_title="Retail SLU Inference", page_icon="🎧", layout="wide")
st.title("Retail Speech Understanding – Inference Demo")
st.caption("Pilih checkpoint Stage-2 dan jalankan inferensi audio retail secara lokal.")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_data(show_spinner=False)
def load_label_maps_cached(path: Path) -> LabelMaps:
    return LabelMaps.from_json(path)


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: Path) -> torch.nn.Module:
    maps = load_label_maps_cached(LABEL_MAP_PATH)
    model = load_model(model_path, maps, device=DEVICE)
    return model.to(DEVICE)


def ensure_artifacts() -> Optional[str]:
    if not LABEL_MAP_PATH.exists():
        return f"Label map tidak ditemukan di {LABEL_MAP_PATH}"
    if not any(path.exists() for path in MODEL_CHOICES.values()):
        choices_str = "\n".join(path.as_posix() for path in MODEL_CHOICES.values())
        return (
            "Tidak menemukan checkpoint Stage-2 di lokasi berikut:\n"
            f"{choices_str}\nPeriksa kembali nama file atau jalankan ulang training Stage-2."
        )
    return None


artifact_error = ensure_artifacts()
if artifact_error:
    st.error(artifact_error)
    st.stop()

label_maps = load_label_maps_cached(LABEL_MAP_PATH)

available_models = {name: path for name, path in MODEL_CHOICES.items() if path.exists()}
missing_models = [name for name in MODEL_CHOICES if name not in available_models]

st.sidebar.header("Konfigurasi Model")
if missing_models:
    st.sidebar.warning(
        "Checkpoint berikut belum tersedia:\n" + "\n".join(MODEL_CHOICES[name].as_posix() for name in missing_models)
    )

choice_labels = list(available_models.keys())
default_index = min(1, len(choice_labels) - 1) if choice_labels else 0
model_choice = st.sidebar.radio("Pilih checkpoint Stage-2", choice_labels, index=default_index)
selected_model_path = available_models[model_choice]
st.sidebar.caption(selected_model_path.as_posix())

model = load_model_cached(selected_model_path)


def predict_from_features(features: AudioProcessingResult) -> dict[str, object]:
    tensor, lengths = logmel_to_tensor(features.logmel)
    tensor = tensor.to(DEVICE)
    lengths = lengths.to(DEVICE)
    with torch.no_grad():
        output = model(tensor, lengths)
        product_logits = output["product"].cpu().numpy()[0]
        quantity_logits = output["quantity"].cpu().numpy()[0]
        intent_logits = output["intent"].cpu().numpy()[0]

    product_idx = int(np.argmax(product_logits))
    quantity_idx = int(np.argmax(quantity_logits))
    intent_idx = int(np.argmax(intent_logits))
    decoded = label_maps.decode(product_idx, quantity_idx, intent_idx)
    return {
        "decoded": decoded,
        "scores": {
            "product": product_logits,
            "quantity": quantity_logits,
            "intent": intent_logits,
        },
        "frames": features.frame_count,
        "duration": features.duration,
    }


def render_prediction(result: dict[str, object]) -> None:
    decoded = result["decoded"]  # type: ignore[index]
    st.subheader("Prediksi")
    st.metric("Intent", decoded["intent"])
    st.metric("Produk", decoded["product"])
    st.metric("Quantity", decoded["quantity"])

    with st.expander("Detail Logits"):
        scores = result["scores"]  # type: ignore[index]
        st.json({
            "product": {label: float(scores["product"][idx]) for label, idx in label_maps.product.items()},
            "quantity": {label: float(scores["quantity"][idx]) for label, idx in label_maps.quantity.items()},
            "intent": {label: float(scores["intent"][idx]) for label, idx in label_maps.intent.items()},
        })

    meta_col1, meta_col2 = st.columns(2)
    meta_col1.write(f"Durasi: {result['duration']:.2f}s")  # type: ignore[index]
    meta_col2.write(f"Jumlah frame log-Mel: {result['frames']}")  # type: ignore[index]


def _ensure_bytes(data: bytes | memoryview | bytearray | np.ndarray) -> bytes:
    if isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        # st_audiorec returns float32 waveform normalised [-1,1]
        import soundfile as sf  # local import to avoid heavy dependency on startup

        sf.write(buffer, data, RECORDER_SAMPLE_RATE, format="WAV")
        return buffer.getvalue()
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytearray):
        return bytes(data)
    return data


def handle_inference(audio_payload: bytes | memoryview | bytearray, label: str, mime: str = "audio/wav") -> None:
    audio_bytes = _ensure_bytes(audio_payload)
    try:
        with st.spinner("Memproses audio..."):
            features = build_features(audio_bytes)
    except Exception as exc:  # pragma: no cover
        st.error(f"Gagal menyiapkan fitur audio ({label}): {exc}")
        return

    st.audio(audio_bytes, format=mime)

    if st.checkbox("Simpan salinan audio", key=f"save_{label}"):
        timestamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        target_path = UPLOAD_DIR / f"{label}_{timestamp}.wav"
        try:
            save_waveform(target_path, features.waveform)
            st.success(f"Audio tersimpan di {target_path}")
        except Exception as exc:  # pragma: no cover
            st.error(f"Tidak dapat menyimpan audio: {exc}")

    start = time.perf_counter()
    with st.spinner("Menjalankan inferensi..."):
        result = predict_from_features(features)
    elapsed = time.perf_counter() - start
    render_prediction(result)
    st.caption(f"Waktu inferensi: {elapsed:.2f} detik")


upload_tab, record_tab = st.tabs(["Upload Audio", "Rekam Langsung"])

with upload_tab:
    st.write("Unggah file audio (mp3/wav/flac/m4a) dengan durasi 0.5–5 detik.")
    uploaded = st.file_uploader("Pilih file audio", type=["wav", "mp3", "flac", "m4a"], accept_multiple_files=False)
    if uploaded is not None:
        audio_bytes = uploaded.read()
        mime_type = uploaded.type or "audio/wav"
        handle_inference(audio_bytes, label=uploaded.name, mime=mime_type)

with record_tab:
    st.write("Rekam audio langsung di browser. Instalasi paket tambahan mungkin dibutuhkan (contoh: `streamlit-audiorec`).")
    try:
        from st_audiorec import st_audiorec  # type: ignore
    except Exception:  # pragma: no cover
        st.info(
            "Modul `streamlit-audiorec` belum terpasang. Jalankan `pip install streamlit-audiorec` lalu restart aplikasi untuk fitur rekam."
        )
        st.stop()

    wav_audio = st_audiorec()
    if wav_audio is not None:
        st.success("Rekaman diterima. Tekan tombol di bawah untuk menjalankan prediksi.")
        if st.button("Prediksi Rekaman", type="primary"):
            handle_inference(wav_audio, label="recording", mime="audio/wav")
