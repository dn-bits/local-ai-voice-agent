"""LiveKit Agents worker.

Moved verbatim from ``livekit_agent/src/agent.py``. The only change is that the
default base URLs are loopback (``127.0.0.1``) instead of Docker service names —
the supervisor spawns the inference children on loopback ports, so this is
correct for both single-image deployment and bare-metal local runs.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    RunContext,
    cli,
    function_tool,
)
from livekit.plugins import elevenlabs, openai, silero

_TURN_DETECTION = os.getenv("TURN_DETECTION", "multilingual").strip().lower()
if _TURN_DETECTION == "multilingual":
    from livekit.plugins.turn_detector.multilingual import MultilingualModel
else:
    MultilingualModel = None  # type: ignore[misc,assignment]

logger = logging.getLogger("agent")

load_dotenv(".env.local")


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a helpful voice AI assistant. The user is interacting with you via "
                "voice, even if you perceive the conversation as text. You eagerly assist "
                "users with their questions by providing information from your extensive "
                "knowledge. Your responses are concise, to the point, and without any "
                "emojis, lists, or other special symbols. "
                "Use short sentences so spoken audio can begin quickly. "
                "You are curious, friendly, and have a sense of humor."
            ),
        )

    @function_tool()
    async def multiply_numbers(
        self,
        context: RunContext,
        number1: int,
        number2: int,
    ) -> dict[str, Any]:
        """Multiply two numbers.

        Args:
            number1: The first number to multiply.
            number2: The second number to multiply.
        """
        return f"The product of {number1} and {number2} is {number1 * number2}."


_idle_processes = os.getenv("AGENT_IDLE_PROCESSES")
server = AgentServer(**({"num_idle_processes": int(_idle_processes)} if _idle_processes else {}))


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


def _turn_detection_mode():
    # Passing None disables automatic turn completion in current LiveKit
    # Agents. The low-memory profile needs explicit VAD-only endpointing.
    return "vad"


def _stt_language() -> str:
    return os.getenv("STT_LANGUAGE", "en")


@server.rtc_session(agent_name="local-voice-ai-elevenlabs")
async def my_agent(ctx: JobContext) -> None:
    ctx.log_context_fields = {"room": ctx.room.name}
    logger.info("MY_AGENT CALLBACK INVOKED: room=%s", ctx.room.name)

    llama_model = os.getenv("LLAMA_MODEL", "gemma-4-e2b")
    llama_base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    llama_api_key = os.getenv("LLAMA_API_KEY", "no-key-needed")

    stt_provider = os.getenv("STT_PROVIDER", "nemotron-cpp").lower()
    if stt_provider == "whisper":
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "Systran/faster-whisper-small"
    else:
        default_stt_base_url = "http://127.0.0.1:8000/v1"
        default_stt_model = "nemotron-speech-streaming"

    stt_base_url = os.getenv("STT_BASE_URL", default_stt_base_url)
    stt_model = os.getenv("STT_MODEL", default_stt_model)
    stt_api_key = os.getenv("STT_API_KEY", "no-key-needed")

    tts_provider = os.getenv("TTS_PROVIDER", "elevenlabs").lower()
    elevenlabs_voice_id = os.getenv(
        "ELEVENLABS_VOICE_ID",
        "fPIfC3elMLbN9tNwMXkw",
    )
    elevenlabs_model = os.getenv(
        "ELEVENLABS_MODEL_ID",
        "eleven_multilingual_v2",
    )
    elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

    tts = elevenlabs.TTS(
        voice_id=elevenlabs_voice_id,
        model=elevenlabs_model,
        api_key=elevenlabs_api_key,
    )

    logger.info(
        "agent session: stt=%s/%s llm=%s/%s tts=%s",
        stt_provider,
        stt_model,
        llama_base_url,
        llama_model,
        tts_provider,
    )

    wake_word = os.getenv("WAKE_WORD", "").strip().lower() in {"1", "true", "yes", "on"}
    wake_word_model = os.getenv("WAKE_WORD_MODEL", "/app/models/wakeword/hey_livekit.onnx")
    wake_word_threshold = float(os.getenv("WAKE_WORD_THRESHOLD", "0.5"))

    session_vad = ctx.proc.userdata["vad"]

    if stt_provider == "nemotron-cpp":
        from .nemotron_stt import NemotronSTT

        stt = NemotronSTT(
            base_url=stt_base_url,
            model=stt_model,
            api_key=stt_api_key,
            language=_stt_language(),
            endpointing_ms=int(os.getenv("NEMOTRON_ENDPOINTING_MS", "300")),
            # NeMo-Speech.cpp requires the client to commit each utterance.
            # Reuse Silero's weights in a second stream so real microphone
            # noise does not make endpointing depend on a fragile RMS cutoff.
            vad_model=session_vad,
        )
    else:
        stt = openai.STT(base_url=stt_base_url, model=stt_model, api_key=stt_api_key)

    session = AgentSession(
        stt=stt,
        llm=openai.LLM(base_url=llama_base_url, model=llama_model, api_key=llama_api_key),
        tts=elevenlabs.TTS(
            voice_id=elevenlabs_voice_id,
            model=elevenlabs_model,
            api_key=elevenlabs_api_key,
        ),
        turn_detection=_turn_detection_mode(),
        vad=session_vad,
        preemptive_generation=True,
    )

    await session.start(agent=Assistant(), room=ctx.room)
    await ctx.connect()

    if wake_word:
        # Join deaf, wait for the wake phrase, then wake up and greet.
        from .wakeword import wait_for_wake_word

        session.input.set_audio_enabled(False)
        participant = await ctx.wait_for_participant()
        try:
            await wait_for_wake_word(participant, wake_word_model, wake_word_threshold)
        except Exception:
            # Fail open: a broken detector shouldn't brick the assistant.
            logger.exception("wake word detection failed; enabling audio input")
        session.input.set_audio_enabled(True)
        session.generate_reply(
            instructions=(
                "You just woke up because the user said the wake phrase. "
                "Greet them very briefly and ask how you can help."
            )
        )
    else:
        # Speak first so the user knows the audio path works.
        session.generate_reply(
            instructions=(
                "Greet the user warmly in one short sentence and invite them to ask you anything."
            )
        )


if __name__ == "__main__":
    cli.run_app(server)
