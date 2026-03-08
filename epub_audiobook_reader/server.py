#!/usr/bin/env python3
"""
EPUB Audiobook Reader — powered by Faster Qwen3-TTS

Usage:
    epub-audiobook-reader
    epub-audiobook-reader --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --port 7861
    epub-audiobook-reader --no-preload
"""

import argparse
import asyncio
import base64
import hashlib
import io
import json
import os
import re
import ssl
import sys
import tempfile
import threading
import time
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

try:
    from faster_qwen3_tts import FasterQwen3TTS
except ImportError:
    print("Error: faster_qwen3_tts not found.")
    print("Install with:  pip install faster-qwen3-tts")
    sys.exit(1)

try:
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup
except ImportError:
    print("Error: ebooklib or beautifulsoup4 not found.")
    print("Install with:  pip install ebooklib beautifulsoup4")
    sys.exit(1)

_QWEN3_TTS_ROOT = os.environ.get("QWEN3_TTS_ROOT", "")

_ALL_MODELS = [
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
]

_active_models_env = os.environ.get("ACTIVE_MODELS", "")
if _active_models_env:
    _allowed = {m.strip() for m in _active_models_env.split(",") if m.strip()}
    AVAILABLE_MODELS = [m for m in _ALL_MODELS if m in _allowed]
else:
    AVAILABLE_MODELS = list(_ALL_MODELS)

BASE_DIR = Path(__file__).resolve().parent
_ASSET_DIR = Path(os.environ.get("ASSET_DIR", "/tmp/faster-qwen3-tts-assets"))
PRESET_TRANSCRIPTS = _ASSET_DIR / "samples" / "parity" / "icl_transcripts.txt"
_LOCAL_REFS_DIR = Path(os.environ.get("LOCAL_REFS_DIR", str(BASE_DIR.parent)))
PRESET_REFS = [
    ("ref_audio_3", _ASSET_DIR / "ref_audio_3.wav", "Clone 1"),
    ("ref_audio_2", _ASSET_DIR / "ref_audio_2.wav", "Clone 2"),
    ("ref_audio", _ASSET_DIR / "ref_audio.wav", "Clone 3"),
]

_GITHUB_RAW = "https://raw.githubusercontent.com/andimarafioti/faster-qwen3-tts/main"
_PRESET_REMOTE = {
    "ref_audio":   f"{_GITHUB_RAW}/ref_audio.wav",
    "ref_audio_2": f"{_GITHUB_RAW}/ref_audio_2.wav",
    "ref_audio_3": f"{_GITHUB_RAW}/ref_audio_3.wav",
}
_TRANSCRIPT_REMOTE = f"{_GITHUB_RAW}/samples/parity/icl_transcripts.txt"


def _fetch_preset_assets() -> None:
    import urllib.request
    _ASSET_DIR.mkdir(parents=True, exist_ok=True)
    PRESET_TRANSCRIPTS.parent.mkdir(parents=True, exist_ok=True)
    if not PRESET_TRANSCRIPTS.exists():
        try:
            urllib.request.urlretrieve(_TRANSCRIPT_REMOTE, PRESET_TRANSCRIPTS)
        except Exception as e:
            print(f"Warning: could not fetch transcripts: {e}")
    for key, path, _ in PRESET_REFS:
        if not path.exists() and key in _PRESET_REMOTE:
            try:
                urllib.request.urlretrieve(_PRESET_REMOTE[key], path)
                print(f"Downloaded {path.name}")
            except Exception as e:
                print(f"Warning: could not fetch {key}: {e}")


_preset_refs: dict[str, dict] = {}


def _load_preset_transcripts() -> dict[str, str]:
    if not PRESET_TRANSCRIPTS.exists():
        return {}
    transcripts = {}
    for line in PRESET_TRANSCRIPTS.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key_part, text = line.split(":", 1)
        key = key_part.split("(")[0].strip()
        transcripts[key] = text.strip()
    return transcripts


def _load_preset_refs() -> None:
    transcripts = _load_preset_transcripts()
    for key, path, label in PRESET_REFS:
        if not path.exists():
            continue
        content = path.read_bytes()
        cached_path = _get_cached_ref_path(content)
        _preset_refs[key] = {
            "id": key,
            "label": label,
            "filename": path.name,
            "path": cached_path,
            "ref_text": transcripts.get(key, ""),
        }


# ─── Model state ─────────────────────────────────────────────────────────────

_model_cache: OrderedDict[str, FasterQwen3TTS] = OrderedDict()
_model_cache_max: int = int(os.environ.get("MODEL_CACHE_SIZE", "2"))
_active_model_name: str | None = None
_loading = False
_generation_lock = asyncio.Lock()
_chapter_cancel_event: threading.Event | None = None
_chapter_active_gen = [None]  # mutable ref to current streaming generator

# ─── Ref audio cache ─────────────────────────────────────────────────────────

_ref_cache: dict[str, str] = {}
_ref_cache_lock = threading.Lock()


def _get_cached_ref_path(content: bytes) -> str:
    digest = hashlib.sha1(content).hexdigest()
    with _ref_cache_lock:
        cached = _ref_cache.get(digest)
        if cached and os.path.exists(cached):
            return cached
        tmp_dir = Path(tempfile.gettempdir())
        path = tmp_dir / f"faster_qwen3_tts_ref_{digest}.wav"
        if not path.exists():
            path.write_bytes(content)
        _ref_cache[digest] = str(path)
        return str(path)


# ─── EPUB state ──────────────────────────────────────────────────────────────

_epub_chapters: list[dict] = []  # [{title, segments: [str]}]


def _detect_language(text: str) -> str:
    """Detect if text is primarily Chinese or English."""
    chinese_count = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
    total = max(len(text.replace(" ", "")), 1)
    return "Chinese" if chinese_count / total > 0.3 else "English"


PAUSE_MARKER = "__PAUSE__"


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs by blank lines. Returns list of paragraph strings."""
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences at Chinese/English sentence boundaries."""
    # First fix missing spaces: "monitor.The" -> "monitor. The"
    text = re.sub(r'([.!?])([A-Z"\u201c])', r'\1 \2', text)
    # Split after Chinese sentence-ending punctuation (。！？；)
    # and after English sentence-ending punctuation followed by space + uppercase
    parts = re.split(
        r'(?<=[。！？；\uFF01\uFF1F])'   # after Chinese punctuation
        r'|(?<=[.!?])(?=\s+[A-Z])',       # after English punctuation + space + uppercase
        text,
    )
    return [p.strip() for p in parts if p.strip()]


def _chunk_sentences(sentences: list[str], max_chars: int) -> list[str]:
    """Group sentences into chunks up to max_chars."""
    chunks = []
    current = ""
    for sent in sentences:
        if current and len(current) + len(sent) > max_chars:
            chunks.append(current.strip())
            current = sent
        else:
            current = (current + sent) if current else sent
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _hard_split_word_boundary(text: str, max_chars: int) -> list[str]:
    """Hard split at word/character boundaries to avoid cutting mid-word."""
    if len(text) <= max_chars:
        return [text]
    result = []
    while len(text) > max_chars:
        # Find the last space within max_chars
        cut = text.rfind(' ', 0, max_chars)
        if cut <= 0:
            # No space found — check for CJK text (can cut anywhere)
            cut = max_chars
        result.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        result.append(text)
    return result


def _force_split(text: str, max_chars: int) -> list[str]:
    """Force-split text that's still too long, trying commas/colons first, then word boundaries."""
    if len(text) <= max_chars:
        return [text]
    # Try splitting at commas, colons, semicolons (Chinese and English)
    parts = re.split(r'(?<=[，,;:；：])', text)
    if len(parts) > 1:
        chunks = _chunk_sentences(parts, max_chars)
        result = []
        for c in chunks:
            if len(c) > max_chars:
                result.extend(_hard_split_word_boundary(c, max_chars))
            else:
                result.append(c)
        return result
    # No punctuation to split on — split at word boundaries
    return _hard_split_word_boundary(text, max_chars)


def _guess_max_chars(text: str) -> int:
    """Return a sensible max_chars based on text language.
    CJK text: 100 chars.  Latin/English text: 250 chars."""
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf')
    return 100 if cjk > len(text) * 0.3 else 250


def _chunk_long_paragraph(text: str, max_chars: int = 100) -> list[str]:
    """Split an overly long paragraph at sentence boundaries. Keep it intact if short enough."""
    if len(text) <= max_chars:
        return [text]
    sentences = _split_sentences(text)
    chunks = _chunk_sentences(sentences, max_chars)
    if not chunks:
        return _force_split(text, max_chars)
    # Force-split any chunks that are still too long
    result = []
    for c in chunks:
        if len(c) > max_chars:
            result.extend(_force_split(c, max_chars))
        else:
            result.append(c)
    return result


def _build_segments_with_pauses(text: str) -> list[str]:
    """Split text into segments by paragraph, inserting PAUSE_MARKER between them."""
    # Fix missing spaces between sentences: "monitor.The" -> "monitor. The"
    text = re.sub(r'([.!?])([A-Z"\u201c])', r'\1 \2', text)
    paragraphs = _split_into_paragraphs(text)
    max_chars = _guess_max_chars(text)
    segments = []
    for i, para in enumerate(paragraphs):
        chunks = _chunk_long_paragraph(para, max_chars=max_chars)
        segments.extend(chunks)
        if i < len(paragraphs) - 1 and chunks:
            segments.append(PAUSE_MARKER)
    return segments


def _parse_epub(content: bytes) -> list[dict]:
    """Parse epub bytes into chapters with text segments."""
    tmp = tempfile.NamedTemporaryFile(suffix=".epub", delete=False)
    tmp.write(content)
    tmp.close()

    try:
        book = epub.read_epub(tmp.name, options={"ignore_ncx": True})
    finally:
        os.unlink(tmp.name)

    chapters = []
    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        # Insert double newlines after block-level elements to preserve paragraph boundaries
        for tag in soup.find_all(["p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "br", "li", "blockquote"]):
            tag.insert_after("\n\n")
        text = soup.get_text(separator=" ")
        # Clean up whitespace but preserve blank lines as paragraph boundaries
        lines = [line.strip() for line in text.splitlines()]
        cleaned = []
        prev_blank = False
        for line in lines:
            if not line:
                if not prev_blank:
                    cleaned.append("")
                prev_blank = True
            else:
                cleaned.append(line)
                prev_blank = False
        text = "\n".join(cleaned).strip()
        # Fix missing spaces between sentences
        text = re.sub(r'([.!?])([A-Z"\u201c])', r'\1 \2', text)

        if not text.strip() or len(text.strip()) < 10:
            continue

        # Try to extract title from headings
        heading = soup.find(["h1", "h2", "h3"])
        title = heading.get_text(strip=True) if heading else None
        if not title:
            # Use first line as title, truncated
            first_line = text.strip().split("\n")[0][:50]
            title = first_line

        segments = _build_segments_with_pauses(text)

        if segments:
            chapters.append({"title": title, "segments": segments})

    return chapters


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _to_wav_b64(audio: np.ndarray, sr: int) -> str:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if audio.ndim > 1:
        audio = audio.squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode()


def _concat_audio(audio_list) -> np.ndarray:
    if isinstance(audio_list, np.ndarray):
        return audio_list.astype(np.float32).squeeze()
    parts = [np.array(a, dtype=np.float32).squeeze() for a in audio_list if len(a) > 0]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def _resolve_model_path(model_id: str) -> str:
    if _QWEN3_TTS_ROOT:
        name = model_id.split("/")[-1] if "/" in model_id else model_id
        local = os.path.join(_QWEN3_TTS_ROOT, name)
        if os.path.isdir(local):
            return local
    return model_id


# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="EPUB Audiobook Reader")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


_fetch_preset_assets()
_load_preset_refs()


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/status")
async def get_status():
    speakers = []
    model_type = None
    active = _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None:
        try:
            model_type = active.model.model.tts_model_type
            speakers = active.model.get_supported_speakers() or []
        except Exception:
            speakers = []
    return {
        "loaded": active is not None,
        "model": _active_model_name,
        "loading": _loading,
        "available_models": AVAILABLE_MODELS,
        "model_type": model_type,
        "speakers": speakers,
        "preset_refs": [
            {"id": p["id"], "label": p["label"], "ref_text": p["ref_text"]}
            for p in _preset_refs.values()
        ],
    }


@app.post("/custom_voice")
async def add_custom_voice(
    label: str = Form(...),
    ref_text: str = Form(""),
    ref_audio: UploadFile = File(...),
):
    """Add a custom voice preset at runtime."""
    content = await ref_audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty audio file")

    voice_id = label.lower().replace(" ", "_")
    cached_path = _get_cached_ref_path(content)

    _preset_refs[voice_id] = {
        "id": voice_id,
        "label": label,
        "filename": ref_audio.filename or f"{voice_id}.wav",
        "path": cached_path,
        "ref_text": ref_text,
        "audio_b64": base64.b64encode(content).decode(),
    }

    # Pre-warm the voice-clone cache for the active model
    active = _model_cache.get(_active_model_name) if _active_model_name else None
    if active is not None and ref_text:
        def _warm():
            for xvec_only in (True, False):
                try:
                    active._prepare_generation(
                        text="Hello.",
                        ref_audio=cached_path,
                        ref_text=ref_text,
                        language="English",
                        xvec_only=xvec_only,
                        non_streaming_mode=True,
                    )
                except Exception:
                    continue
        await asyncio.to_thread(_warm)

    return {"status": "added", "id": voice_id, "label": label}


@app.delete("/custom_voice/{voice_id}")
async def remove_custom_voice(voice_id: str):
    """Remove a custom voice preset."""
    if voice_id not in _preset_refs:
        raise HTTPException(status_code=404, detail="Voice not found")
    _preset_refs.pop(voice_id)
    return {"status": "removed", "id": voice_id}


@app.post("/load")
async def load_model(model_id: str = Form(...)):
    global _active_model_name, _loading

    if model_id in _model_cache:
        _active_model_name = model_id
        _model_cache.move_to_end(model_id)
        return {"status": "already_loaded", "model": model_id}

    _loading = True

    def _do_load():
        global _active_model_name, _loading
        try:
            # Evict ALL old models to free GPU memory
            while _model_cache:
                evicted, old_model = _model_cache.popitem(last=False)
                del old_model
                print(f"Evicted model: {evicted}")
            torch.cuda.empty_cache()
            resolved = _resolve_model_path(model_id)
            new_model = FasterQwen3TTS.from_pretrained(
                resolved, device="cuda", dtype=torch.bfloat16,
            )
            print("Capturing CUDA graphs…")
            new_model._warmup(prefill_len=100)
            _model_cache[model_id] = new_model
            _model_cache.move_to_end(model_id)
            _active_model_name = model_id
            print("CUDA graphs captured — model ready.")
        finally:
            _loading = False

    async with _generation_lock:
        await asyncio.to_thread(_do_load)
    return {"status": "loaded", "model": model_id}


@app.post("/upload")
async def upload_epub(file: UploadFile = File(...)):
    global _epub_chapters
    content = await file.read()
    _epub_chapters = _parse_epub(content)

    if not _epub_chapters:
        raise HTTPException(status_code=400, detail="No readable chapters found in this EPUB.")

    # Return chapter list with segment counts and text
    result = []
    for i, ch in enumerate(_epub_chapters):
        result.append({
            "index": i,
            "title": ch["title"],
            "segment_count": len(ch["segments"]),
            "segments": ch["segments"],
        })
    return {"chapters": result}


@app.post("/generate_chapter")
async def generate_chapter(
    chapter_index: int = Form(...),
    start_segment: int = Form(0),
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    ref_preset: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    """Stream audio for all segments in a chapter (from start_segment onwards).

    SSE events:
      - segment_start: {segment_index}
      - chunk: {audio_b64, sample_rate, segment_index}
      - segment_done: {segment_index, audio_s, gen_ms}
      - done: {total_audio_s, total_ms}
      - error: {message}
    """
    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded.")

    if chapter_index >= len(_epub_chapters) or chapter_index < 0:
        raise HTTPException(status_code=400, detail="Invalid chapter index.")

    chapter = _epub_chapters[chapter_index]
    start_segment = max(0, min(start_segment, len(chapter["segments"]) - 1))

    tmp_path = None

    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        tmp_path = _get_cached_ref_path(content)

    if mode == "voice_clone" and not tmp_path:
        raise HTTPException(status_code=400, detail="Reference audio is required for voice clone mode. Please select a preset or upload a reference audio file.")

    cancel_event = threading.Event()

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    def run_generation():
        try:
            model = _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded.")

            t0 = time.perf_counter()
            total_audio_s = 0.0

            for seg_idx in range(start_segment, len(chapter["segments"])):
                if cancel_event.is_set():
                    return

                text = chapter["segments"][seg_idx]

                # Pause marker: insert 0.8s silence instead of TTS
                if text == PAUSE_MARKER:
                    sr = 24000
                    silence = np.zeros(int(sr * 0.8), dtype=np.float32)
                    total_audio_s += 0.8
                    payload = {
                        "type": "chunk",
                        "audio_b64": _to_wav_b64(silence, sr),
                        "sample_rate": sr,
                        "segment_index": seg_idx,
                    }
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                        "type": "segment_done",
                        "segment_index": seg_idx,
                        "audio_s": 0.8,
                        "gen_ms": 0,
                    }))
                    continue

                language = _detect_language(text)

                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "segment_start",
                    "segment_index": seg_idx,
                    "total_segments": len(chapter["segments"]),
                }))

                seg_t0 = time.perf_counter()
                seg_audio_s = 0.0

                if mode == "voice_clone":
                    gen = model.generate_voice_clone_streaming(
                        text=text, language=language,
                        ref_audio=tmp_path, ref_text=ref_text,
                        xvec_only=True, chunk_size=8,
                        temperature=0.9, top_k=50,
                        repetition_penalty=1.05, max_new_tokens=360,
                    )
                elif mode == "custom":
                    if not speaker:
                        raise ValueError("Speaker ID is required for custom voice")
                    gen = model.generate_custom_voice_streaming(
                        text=text, speaker=speaker, language=language,
                        chunk_size=8, temperature=0.9, top_k=50,
                        repetition_penalty=1.05, max_new_tokens=360,
                    )
                else:
                    raise ValueError(f"Unsupported mode: {mode}")

                _chapter_active_gen[0] = gen
                try:
                    for audio_chunk, sr, timing in gen:
                        if cancel_event.is_set():
                            return
                        audio_chunk = _concat_audio(audio_chunk)
                        dur = len(audio_chunk) / sr
                        seg_audio_s += dur
                        total_audio_s += dur
                        payload = {
                            "type": "chunk",
                            "audio_b64": _to_wav_b64(audio_chunk, sr),
                            "sample_rate": sr,
                            "segment_index": seg_idx,
                        }
                        loop.call_soon_threadsafe(queue.put_nowait, json.dumps(payload))
                finally:
                    _chapter_active_gen[0] = None

                if cancel_event.is_set():
                    return

                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                    "type": "segment_done",
                    "segment_index": seg_idx,
                    "audio_s": round(seg_audio_s, 3),
                    "gen_ms": round((time.perf_counter() - seg_t0) * 1000),
                }))

            loop.call_soon_threadsafe(queue.put_nowait, json.dumps({
                "type": "done",
                "total_audio_s": round(total_audio_s, 3),
                "total_ms": round((time.perf_counter() - t0) * 1000),
            }))

        except GeneratorExit:
            return
        except Exception as e:
            if cancel_event.is_set():
                return
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def sse():
        thread = None
        try:
            await _generation_lock.acquire()
            # Set global cancel event AFTER acquiring lock, so cancel_chapter()
            # targets the correct event (not a stale one from a queued request).
            global _chapter_cancel_event
            _chapter_cancel_event = cancel_event
            thread = threading.Thread(target=run_generation, daemon=True)
            thread.start()

            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            cancel_event.set()
        finally:
            cancel_event.set()
            # Close the active generator to interrupt GPU inference immediately
            gen = _chapter_active_gen[0]
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            # Wait for generation thread to actually stop before releasing lock
            if thread is not None:
                await asyncio.to_thread(thread.join, timeout=15)
            if _generation_lock.locked():
                _generation_lock.release()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/cancel_chapter")
async def cancel_chapter():
    if _chapter_cancel_event:
        _chapter_cancel_event.set()
    # Close the active generator to interrupt GPU inference immediately
    gen = _chapter_active_gen[0]
    if gen is not None:
        try:
            gen.close()
        except Exception:
            pass
    return {"status": "cancelling"}


# ─── Audiobook generation ────────────────────────────────────────────────────

_audiobook_output: str | None = None  # path to generated file
_audiobook_cancel_event: threading.Event | None = None
_audiobook_active_gen = [None]


@app.post("/generate_audiobook")
async def generate_audiobook(
    mode: str = Form("voice_clone"),
    ref_text: str = Form(""),
    speaker: str = Form(""),
    ref_preset: str = Form(""),
    chapter_indices: str = Form(""),
    ref_audio: UploadFile = File(None),
):
    global _audiobook_output

    if not _active_model_name or _active_model_name not in _model_cache:
        raise HTTPException(status_code=400, detail="Model not loaded.")
    if not _epub_chapters:
        raise HTTPException(status_code=400, detail="No EPUB loaded.")

    # Parse selected chapter indices
    if chapter_indices:
        try:
            selected = json.loads(chapter_indices)
        except (json.JSONDecodeError, TypeError):
            selected = list(range(len(_epub_chapters)))
    else:
        selected = list(range(len(_epub_chapters)))

    # Validate indices
    selected = [i for i in selected if 0 <= i < len(_epub_chapters)]
    if not selected:
        raise HTTPException(status_code=400, detail="No valid chapters selected.")

    tmp_path = None
    if ref_preset and ref_preset in _preset_refs:
        preset = _preset_refs[ref_preset]
        tmp_path = preset["path"]
        if not ref_text:
            ref_text = preset["ref_text"]
    elif ref_audio and ref_audio.filename:
        content = await ref_audio.read()
        tmp_path = _get_cached_ref_path(content)

    if mode == "voice_clone" and not tmp_path:
        raise HTTPException(status_code=400, detail="Reference audio required for voice clone mode.")

    # Count total segments for selected chapters
    total_segments = sum(len(_epub_chapters[i]["segments"]) for i in selected)

    loop = asyncio.get_event_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()
    cancel_event = threading.Event()

    def run():
        global _audiobook_output
        try:
            model = _model_cache.get(_active_model_name)
            if model is None:
                raise RuntimeError("No model loaded.")

            all_audio = []
            sample_rate = 24000
            seg_done = 0
            t0 = time.perf_counter()

            for ch_idx in selected:
                chapter = _epub_chapters[ch_idx]
                if cancel_event.is_set():
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "cancelled"}))
                    return

                for seg_idx, text in enumerate(chapter["segments"]):
                    if cancel_event.is_set():
                        loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "cancelled"}))
                        return

                    # Pause marker: insert silence
                    if text == PAUSE_MARKER:
                        all_audio.append(np.zeros(int(sample_rate * 0.8), dtype=np.float32))
                        seg_done += 1
                        continue

                    language = _detect_language(text)

                    if mode == "voice_clone":
                        gen = model.generate_voice_clone_streaming(
                            text=text, language=language,
                            ref_audio=tmp_path, ref_text=ref_text,
                            xvec_only=True, chunk_size=8,
                            temperature=0.9, top_k=50,
                            repetition_penalty=1.05, max_new_tokens=360,
                        )
                    elif mode == "custom":
                        gen = model.generate_custom_voice_streaming(
                            text=text, speaker=speaker, language=language,
                            chunk_size=8, temperature=0.9, top_k=50,
                            repetition_penalty=1.05, max_new_tokens=360,
                        )
                    else:
                        raise ValueError(f"Unsupported mode: {mode}")

                    _audiobook_active_gen[0] = gen
                    seg_audio = []
                    try:
                        for audio_chunk, sr, timing in gen:
                            if cancel_event.is_set():
                                loop.call_soon_threadsafe(queue.put_nowait, json.dumps({"type": "cancelled"}))
                                return
                            sample_rate = sr
                            chunk = _concat_audio(audio_chunk)
                            seg_audio.append(chunk)
                    finally:
                        _audiobook_active_gen[0] = None

                    if seg_audio:
                        all_audio.append(np.concatenate(seg_audio))

                    seg_done += 1
                    elapsed = time.perf_counter() - t0
                    total_audio = sum(len(a) for a in all_audio) / sample_rate
                    progress = {
                        "type": "progress",
                        "segment": seg_done,
                        "total": total_segments,
                        "chapter": ch_idx,
                        "chapter_title": chapter["title"],
                        "pct": round(seg_done / total_segments * 100, 1),
                        "elapsed_s": round(elapsed, 1),
                        "audio_s": round(total_audio, 1),
                    }
                    loop.call_soon_threadsafe(queue.put_nowait, json.dumps(progress))

                # Add 0.8s silence between chapters
                all_audio.append(np.zeros(int(sample_rate * 0.8), dtype=np.float32))

            # Save to file
            full_audio = np.concatenate(all_audio) if all_audio else np.zeros(0, dtype=np.float32)
            out_path = os.path.join(tempfile.gettempdir(), "epub_audiobook.wav")
            sf.write(out_path, full_audio, sample_rate, subtype="PCM_16")
            _audiobook_output = out_path

            total_dur = len(full_audio) / sample_rate
            done = {
                "type": "done",
                "total_audio_s": round(total_dur, 1),
                "total_ms": round((time.perf_counter() - t0) * 1000),
                "download_url": "/download_audiobook",
            }
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(done))

        except GeneratorExit:
            return
        except Exception as e:
            if cancel_event.is_set():
                return
            import traceback
            err = {"type": "error", "message": str(e), "detail": traceback.format_exc()}
            loop.call_soon_threadsafe(queue.put_nowait, json.dumps(err))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    async def sse():
        thread = None
        try:
            await _generation_lock.acquire()
            global _audiobook_cancel_event
            _audiobook_cancel_event = cancel_event
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            while True:
                msg = await queue.get()
                if msg is None:
                    break
                yield f"data: {msg}\n\n"
        except asyncio.CancelledError:
            cancel_event.set()
        finally:
            cancel_event.set()
            gen = _audiobook_active_gen[0]
            if gen is not None:
                try:
                    gen.close()
                except Exception:
                    pass
            if thread is not None:
                await asyncio.to_thread(thread.join, timeout=15)
            if _generation_lock.locked():
                _generation_lock.release()

    return StreamingResponse(
        sse(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/cancel_audiobook")
async def cancel_audiobook():
    if _audiobook_cancel_event:
        _audiobook_cancel_event.set()
    gen = _audiobook_active_gen[0]
    if gen is not None:
        try:
            gen.close()
        except Exception:
            pass
    return {"status": "cancelling"}


@app.get("/download_audiobook")
async def download_audiobook():
    if not _audiobook_output or not os.path.exists(_audiobook_output):
        raise HTTPException(status_code=404, detail="No audiobook generated yet.")
    return FileResponse(
        _audiobook_output,
        media_type="audio/wav",
        filename="audiobook.wav",
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EPUB Audiobook Reader")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Model to preload at startup (default: 1.7B-Base)",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7861)))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="Skip model loading at startup (load via UI instead)",
    )
    parser.add_argument(
        "--ssl",
        action="store_true",
        help="Enable HTTPS with self-signed certificate (required for microphone recording)",
    )
    args = parser.parse_args()

    if not args.no_preload:
        global _active_model_name
        resolved = _resolve_model_path(args.model)
        print(f"Loading model: {args.model} -> {resolved}")
        _startup_model = FasterQwen3TTS.from_pretrained(
            resolved, device="cuda", dtype=torch.bfloat16,
        )
        print("Capturing CUDA graphs…")
        _startup_model._warmup(prefill_len=100)
        _model_cache[args.model] = _startup_model
        _active_model_name = args.model
        print("TTS model ready.")

    # Generate self-signed cert for HTTPS (needed for microphone access)
    ssl_kwargs = {}
    if args.ssl:
        cert_dir = Path(__file__).parent / ".ssl"
        cert_dir.mkdir(exist_ok=True)
        cert_file = cert_dir / "cert.pem"
        key_file = cert_dir / "key.pem"
        if not cert_file.exists() or not key_file.exists():
            print("Generating self-signed SSL certificate...")
            import subprocess
            subprocess.run([
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", str(key_file), "-out", str(cert_file),
                "-days", "365", "-nodes",
                "-subj", "/CN=localhost",
            ], check=True, capture_output=True)
        ssl_kwargs["ssl_certfile"] = str(cert_file)
        ssl_kwargs["ssl_keyfile"] = str(key_file)
        proto = "https"
    else:
        proto = "http"

    print(f"Ready. Open {proto}://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", **ssl_kwargs)


if __name__ == "__main__":
    main()
