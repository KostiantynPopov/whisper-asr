import importlib.metadata
import os
from os import path
from threading import Lock
from typing import Annotated, Optional, Union
from urllib.parse import quote

import click
import uvicorn
from fastapi import FastAPI, File, Query, UploadFile, applications
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from whisper import tokenizer

from app.config import CONFIG
from app.factory.asr_model_factory import ASRModelFactory
from app.utils import load_audio

# Lazy loading: model will be loaded per worker on first request
asr_model = None
asr_model_lock = Lock()

LANGUAGE_CODES = sorted(tokenizer.LANGUAGES.keys())

projectMetadata = importlib.metadata.metadata("whisper-asr-webservice")
app = FastAPI(
    title=projectMetadata["Name"].title().replace("-", " "),
    description=projectMetadata["Summary"],
    version=projectMetadata["Version"],
    contact={"url": projectMetadata["Home-page"]},
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    license_info={"name": "MIT License", "url": "https://github.com/ahmetoner/whisper-asr-webservice/blob/main/LICENCE"},
)

assets_path = os.getcwd() + "/swagger-ui-assets"
if path.exists(assets_path + "/swagger-ui.css") and path.exists(assets_path + "/swagger-ui-bundle.js"):
    app.mount("/assets", StaticFiles(directory=assets_path), name="static")

    def swagger_monkey_patch(*args, **kwargs):
        return get_swagger_ui_html(
            *args,
            **kwargs,
            swagger_favicon_url="",
            swagger_css_url="/assets/swagger-ui.css",
            swagger_js_url="/assets/swagger-ui-bundle.js",
        )

    applications.get_swagger_ui_html = swagger_monkey_patch


@app.get("/", response_class=RedirectResponse, include_in_schema=False)
async def index():
    return "/docs"


@app.post("/asr", tags=["Endpoints"])
async def asr(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
    task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
    language: Union[str, None] = Query(default=None, enum=LANGUAGE_CODES),
    initial_prompt: Union[str, None] = Query(default=None),
    vad_filter: Annotated[
        bool | None,
        Query(
            description="Enable the voice activity detection (VAD) to filter out parts of the audio without speech",
            include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
        ),
    ] = False,
    word_timestamps: bool = Query(
        default=False,
        description="Word level timestamps",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "faster_whisper" else False),
    ),
    diarize: bool = Query(
        default=False,
        description="Diarize the input",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" and CONFIG.HF_TOKEN != "" else False),
    ),
    min_speakers: Union[int, None] = Query(
        default=None,
        description="Min speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    max_speakers: Union[int, None] = Query(
        default=None,
        description="Max speakers in this file",
        include_in_schema=(True if CONFIG.ASR_ENGINE == "whisperx" else False),
    ),
    initial_offset: Union[float, None] = Query(
        default=None,
        description="Initial silence offset in seconds to add to all timestamps",
    ),
    auto_calculate_offset: bool = Query(
        default=False,
        description="Automatically calculate initial silence offset from audio",
    ),
    output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"]),
):
    global asr_model
    # Lazy loading: load model on first request per worker (thread-safe)
    if asr_model is None:
        with asr_model_lock:
            # Double-check pattern: another thread might have loaded it while we waited
            if asr_model is None:
                asr_model = ASRModelFactory.create_asr_model()
                asr_model.load_model()
    
    result = asr_model.transcribe(
        load_audio(audio_file.file, encode),
        task,
        language,
        initial_prompt,
        vad_filter,
        word_timestamps,
        {
            "diarize": diarize,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
            "initial_offset": initial_offset,
            "auto_calculate_offset": auto_calculate_offset,
        },
        output,
    )
    return StreamingResponse(
        result,
        media_type="text/plain",
        headers={
            "Asr-Engine": CONFIG.ASR_ENGINE,
            "Content-Disposition": f'attachment; filename="{quote(audio_file.filename)}.{output}"',
        },
    )


@app.post("/detect-language", tags=["Endpoints"])
async def detect_language(
    audio_file: UploadFile = File(...),  # noqa: B008
    encode: bool = Query(default=True, description="Encode audio first through FFmpeg"),
):
    global asr_model
    # Lazy loading: load model on first request per worker (thread-safe)
    if asr_model is None:
        with asr_model_lock:
            # Double-check pattern: another thread might have loaded it while we waited
            if asr_model is None:
                asr_model = ASRModelFactory.create_asr_model()
                asr_model.load_model()
    
    detected_lang_code, confidence = asr_model.language_detection(load_audio(audio_file.file, encode))
    return {
        "detected_language": tokenizer.LANGUAGES[detected_lang_code],
        "language_code": detected_lang_code,
        "confidence": confidence,
    }


@click.command()
@click.option(
    "-h",
    "--host",
    metavar="HOST",
    default="0.0.0.0",
    help="Host for the webservice (default: 0.0.0.0)",
)
@click.option(
    "-p",
    "--port",
    metavar="PORT",
    default=9000,
    help="Port for the webservice (default: 9000)",
)
@click.option(
    "-w",
    "--workers",
    metavar="WORKERS",
    default=None,
    type=int,
    help="Number of worker processes (default: from UVICORN_WORKERS env var or 1)",
)
@click.version_option(version=projectMetadata["Version"])
def start(host: str, port: Optional[int] = None, workers: Optional[int] = None):
    # Get workers from environment variable or CLI argument
    num_workers = workers if workers is not None else int(os.getenv("UVICORN_WORKERS", "1"))
    # Pass app as import string to enable workers support
    uvicorn.run("app.webservice:app", host=host, port=port, workers=num_workers)


if __name__ == "__main__":
    start()
