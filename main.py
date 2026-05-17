import os

from dotenv import load_dotenv

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)

from pipecat.runner.run import main
from pipecat.runner.types import SmallWebRTCRunnerArguments

from pipecat.services.groq.llm import GroqLLMService

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


DEFAULT_VOICE_SYSTEM_PROMPT = (
    "You are a helpful assistant in a voice conversation. Your responses will be spoken aloud, "
    "so avoid emojis, bullet points, or other formatting that can't be spoken. Respond to what the "
    "user said in a creative, helpful, and brief way."
)


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

    # Groq LLM
    llm = GroqLLMService(
        api_key=os.getenv("GROQ_API_KEY"),
        settings=GroqLLMService.Settings(
            model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
            system_instruction=os.getenv(
                "VOICE_ASSISTANT_PROMPT",
                DEFAULT_VOICE_SYSTEM_PROMPT,
            ),
        ),
    )

    # Text-to-Speech
    tts = DeepgramTTSService(
        api_key=os.getenv("DEEPGRAM_API_KEY"),
    )

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )

    # Pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
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