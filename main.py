import os

from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.runner.run import main
from pipecat.runner.types import SmallWebRTCRunnerArguments

from pipecat.services.google.llm import GoogleLLMService

from pipecat.services.deepgram.stt import (
    DeepgramSTTService,
)

from pipecat.services.deepgram.tts import (
    DeepgramTTSService,
)

from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)

from pipecat.transports.base_transport import (
    TransportParams,
)

load_dotenv()


def get_client_url() -> str:
    return os.getenv("PUBLIC_CLIENT_URL", "http://localhost:7860/client")


async def bot(runner_args: SmallWebRTCRunnerArguments):

    # WebRTC Transport
    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,

        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_enabled=True,
        ),
    )

    # Speech-to-Text
    stt = DeepgramSTTService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    # Gemini LLM
    llm = GoogleLLMService(
        api_key=os.getenv("GEMINI_API_KEY"),
    )

    # Text-to-Speech
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            llm,
            tts,
            transport.output(),
        ]
    )

    # Task
    task = PipelineTask(pipeline)

    # Runner
    runner = PipelineRunner()

    print("🚀 Voice agent started")
    print(f"🎤 Open: {get_client_url()}")
    print("🗣️ Allow microphone access and speak")

    await runner.run(task)


if __name__ == "__main__":
    main()