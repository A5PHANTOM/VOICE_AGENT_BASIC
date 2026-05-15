import os
import asyncio

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
import uvicorn

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.transports.webrtc.transport import SmallWebRTCTransport

from pipecat_ai_small_webrtc_prebuilt.frontend import (
    SmallWebRTCPrebuiltUI,
)

load_dotenv()

app = FastAPI()

# Mount Pipecat prebuilt WebRTC UI
app.mount("/ui", SmallWebRTCPrebuiltUI)

@app.get("/")
async def root():
    return RedirectResponse("/ui")


async def run_bot():

    transport = SmallWebRTCTransport(
        params={
            "audio_in_enabled": True,
            "audio_out_enabled": True,
        }
    )

    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY")
    )

    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
        model="gemini-1.5-flash",
    )

    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
        voice="aura-asteria-en",
    )

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            llm,
            tts,
            transport.output(),
        ]
    )

    task = PipelineTask(pipeline)

    runner = PipelineRunner()

    print("Voice agent started")

    await runner.run(task)


@app.on_event("startup")
async def startup():
    asyncio.create_task(run_bot())


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
    )