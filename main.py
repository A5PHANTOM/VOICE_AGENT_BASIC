import os

from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask

from pipecat.runner.run import main
from pipecat.runner.types import SmallWebRTCRunnerArguments

from pipecat.services.google.llm import GoogleLLMService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.deepgram.tts import DeepgramTTSService

from pipecat.transports.smallwebrtc.transport import (
    SmallWebRTCTransport,
)

load_dotenv()


async def bot(runner_args: SmallWebRTCRunnerArguments):

    transport = SmallWebRTCTransport(
        webrtc_connection=runner_args.webrtc_connection,
        params={
            "audio_in_enabled": True,
            "audio_out_enabled": True,
        },
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


if __name__ == "__main__":
    main()