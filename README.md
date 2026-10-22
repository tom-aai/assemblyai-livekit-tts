# Quickstart 

This guide covers integrating AssemblyAI's **Streaming TTS** into a LiveKit
voice agent using the [Agents framework](https://docs.livekit.io/agents/) (1.8+).
LiveKit Agents is an open-source framework for building real-time voice agents:
your agent joins a LiveKit room as a participant and exchanges audio with users
over WebRTC or SIP. An `AgentSession` wires together speech-to-text, an LLM, and
text-to-speech, and handles orchestration (turn detection, interruptions).

# Requirements

* **Python 3.10+** and the [`uv`](https://docs.astral.sh/uv/) package manager.
* A **LiveKit Cloud** account ([free tier](https://cloud.livekit.io/)).
* The **LiveKit CLI**, version 2.18.8 or later:
  * **macOS:** `brew install livekit-cli`
  * **Linux:** `curl -sSL https://get.livekit.io/cli | bash`
  * **Windows:** `winget install LiveKit.LiveKitCLI`
* An **AssemblyAI API key** from [API Keys](https://www.assemblyai.com/dashboard/home).

**1. Create a LiveKit agent**

Authenticate the CLI and scaffold a project from LiveKit's Python starter
template, then install its dependencies:

```bash theme={null}
lk cloud auth
lk agent init my-agent --template agent-starter-python
cd my-agent
uv sync
```

`lk cloud auth` also writes your `LIVEKIT_URL`, `LIVEKIT_API_KEY`, and
`LIVEKIT_API_SECRET` into the project, so you don't have to set them by hand.

**2. Add the AssemblyAI TTS plugin**

Clone this repo inside your agent project and install it into the project's
environment as an editable dependency:

```bash theme={null}
git clone https://github.com/tom-aai/assemblyai-livekit-tts.git
uv add --editable ./assemblyai-livekit-tts
```

`uv add --editable` installs the package into your agent's virtual environment,
so `import assemblyai_tts` works from `src/agent.py` regardless of where the
clone lives. Because it's editable, pulling updates in the clone takes effect
without reinstalling.

**3. Set your API key**

The LiveKit credentials are already in `.env.local` from `lk cloud auth`. Add
your AssemblyAI key alongside them:

```env theme={null}
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_livekit_api_key
LIVEKIT_API_SECRET=your_livekit_api_secret
ASSEMBLYAI_API_KEY=your_assemblyai_key
```

**4. Wire the plugin into your agent**

Replace the generated `src/agent.py` with the following. It swaps the template's
default TTS for `assemblyai_tts.TTS`, turns **expressive mode off**, and attaches
a reporter that **emits per-turn latency metrics** to the terminal:

```python expandable theme={null}
import logging
import textwrap

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    ConversationItemAddedEvent,
    JobContext,
    STTContextOptions,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.agents.llm import ChatMessage, MetricsReport
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics
from livekit.plugins import ai_coustics

import assemblyai_tts

logger = logging.getLogger("agent")
logger.setLevel(logging.INFO)

# A dedicated logger for pipeline metrics, so they stand out and can be filtered
# or silenced on their own.
metrics_logger = logging.getLogger("agent.metrics")
metrics_logger.setLevel(logging.INFO)

load_dotenv(".env.local")


# ---------------------------------------------------------------------------
# Pipeline metrics reporting
#
# LiveKit emits latency and usage metrics for each stage of the voice pipeline.
# TurnMetricsReporter logs each component metric as it arrives (STT, LLM, TTS)
# and, once an agent turn is committed, a summary box with the end-of-utterance
# delay, transcription delay, LLM time to first token, TTS time to first byte,
# and the total end-to-end turn latency.
# See https://docs.livekit.io/testing/observability/data/
# ---------------------------------------------------------------------------

_BOX_WIDTH = 48  # total characters per rendered line
_VALUE_WIDTH = 12  # right-aligned column for formatted values


def _fmt_latency(seconds: float | None) -> str:
    """Render a latency value in milliseconds, or an em dash when it's missing."""
    if seconds is None:
        return "—"
    return f"{seconds * 1000:,.1f} ms"


def _fmt_duration(seconds: float | None) -> str:
    """Render a duration value in seconds, or an em dash when it's missing."""
    if seconds is None:
        return "—"
    return f"{seconds:,.2f} s"


def _box_rule(left: str, right: str) -> str:
    return left + "─" * (_BOX_WIDTH - 2) + right


def _box_title(text: str) -> str:
    return f"│ {text:^{_BOX_WIDTH - 4}} │"


def _box_row(label: str, value: str) -> str:
    label_w = _BOX_WIDTH - 4 - _VALUE_WIDTH
    return f"│ {label:<{label_w}}{value:>{_VALUE_WIDTH}} │"


class TurnMetricsReporter:
    """Collects LiveKit pipeline metrics and logs per-component and per-turn latency."""

    def __init__(self) -> None:
        # The end-of-utterance and transcription delays land on the user's message,
        # while the LLM/TTS/end-to-end metrics land on the following assistant message.
        # Hold the most recent user-turn metrics so both halves report together.
        self._pending_user_metrics: MetricsReport = {}

    def attach(self, session: AgentSession) -> None:
        """Subscribe to the session's per-component and conversation events."""
        if session.stt is not None:
            session.stt.on("metrics_collected", self._on_stt_metrics)
        if session.llm is not None:
            session.llm.on("metrics_collected", self._on_llm_metrics)
        if session.tts is not None:
            session.tts.on("metrics_collected", self._on_tts_metrics)
        session.on("conversation_item_added", self._on_conversation_item_added)

    # -- Per-component metrics, logged as each pipeline stage finishes ---------

    def _on_stt_metrics(self, m: STTMetrics) -> None:
        metrics_logger.info(
            "STT  · processing %s · audio %s",
            _fmt_latency(m.duration),
            _fmt_duration(m.audio_duration),
        )

    def _on_llm_metrics(self, m: LLMMetrics) -> None:
        if m.cancelled:
            metrics_logger.info("LLM  · cancelled (interrupted)")
            return
        metrics_logger.info(
            "LLM  · TTFT %s · %d tokens · %.1f tok/s",
            _fmt_latency(m.ttft),
            m.completion_tokens,
            m.tokens_per_second,
        )

    def _on_tts_metrics(self, m: TTSMetrics) -> None:
        if m.cancelled:
            metrics_logger.info("TTS  · cancelled (interrupted)")
            return
        metrics_logger.info(
            "TTS  · TTFB %s · audio %s · %d chars",
            _fmt_latency(m.ttfb),
            _fmt_duration(m.audio_duration),
            m.characters_count,
        )

    # -- Per-turn summary, logged when the agent's reply is committed ----------

    def _on_conversation_item_added(self, ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if not isinstance(item, ChatMessage):
            return
        metrics = item.metrics or {}
        if item.role == "user":
            self._pending_user_metrics = metrics
        elif item.role == "assistant":
            self._log_turn_summary(self._pending_user_metrics, metrics)
            self._pending_user_metrics = {}

    def _log_turn_summary(
        self, user_metrics: MetricsReport, assistant_metrics: MetricsReport
    ) -> None:
        rows = [
            ("End of utterance delay", user_metrics.get("end_of_turn_delay")),
            ("Transcription delay", user_metrics.get("transcription_delay")),
            ("LLM time to first token", assistant_metrics.get("llm_node_ttft")),
            ("TTS time to first byte", assistant_metrics.get("tts_node_ttfb")),
        ]
        rows = [(label, value) for label, value in rows if value is not None]
        e2e_latency = assistant_metrics.get("e2e_latency")
        if not rows and e2e_latency is None:
            return

        lines = [
            "",
            _box_rule("┌", "┐"),
            _box_title("Turn latency summary"),
            _box_rule("├", "┤"),
            *(_box_row(label, _fmt_latency(value)) for label, value in rows),
        ]
        if e2e_latency is not None:
            lines.append(_box_rule("├", "┤"))
            lines.append(_box_row("End-to-end turn latency", _fmt_latency(e2e_latency)))
        lines.append(_box_rule("└", "┘"))
        metrics_logger.info("\n".join(lines))


class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            # See all available models at https://docs.livekit.io/agents/models/llm/
            llm=inference.LLM(model="google/gemma-4-31b-it"),
            instructions=textwrap.dedent(
                """\
                You are a friendly, reliable voice assistant. Respond in plain
                text only, since your output is spoken by a text-to-speech system.
                Keep replies brief: one to three sentences, and ask one question
                at a time. Spell out numbers, phone numbers, and email addresses.
                """
            ),
        )


server = AgentServer()


@server.rtc_session(agent_name="tts-agent")
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        # Speech-to-text via LiveKit Inference (AssemblyAI Universal-3.5 Pro).
        # See https://docs.livekit.io/agents/models/stt/
        stt=inference.STT(model="assemblyai/universal-3-5-pro", language="en"),
        stt_context_options=STTContextOptions(
            keyterms=["LiveKit"],
            keyterm_detection={"enabled": True},
        ),
        # Text-to-speech via the AssemblyAI streaming TTS plugin. Set
        # ASSEMBLYAI_API_KEY in .env.local. Browse voices at /tts/voices; every
        # voice speaks every language.
        tts=assemblyai_tts.TTS(voice_id="aai-alba", language="english"),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
        # Expressive mode injects the TTS provider's markup guide into the LLM
        # prompt so the model emits inline delivery tags (emotion, pacing).
        # It requires a TTS model that supports markup; the AssemblyAI streaming
        # TTS plugin does not, so it is disabled here.
        expressive=False,
    )

    # Collect and print pipeline latency metrics to the terminal. Attach before
    # starting the session so no early turns are missed.
    TurnMetricsReporter().attach(session)

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
```

> [!NOTE]
>
> **Why `expressive=False`.** Expressive mode asks the LLM to emit inline
> markup (emotion, pacing, non-verbal sounds) that the TTS renders. The
> Streaming TTS plugin doesn't yet support that markup, so leaving expressive mode
> on would leak raw tags into the spoken output. Keep it off.

**5. Run and test**

Talk to the agent in your terminal:

```bash theme={null}
lk agent console
```

Each turn prints the component metrics and a latency summary box from
`TurnMetricsReporter`. To connect a frontend or the
[Agent Console](https://docs.livekit.io/testing/agent-console/) instead, run:

```bash theme={null}
lk agent dev
```

#### Plugin options

`assemblyai_tts.TTS(...)` accepts the following, which map onto the Streaming TTS
[connect parameters](/docs/tts/connect-parameters):

**voice_id** `string` (default: aai-jane)

The voice identifier. Preset and library voices are named `aai-<name>` (the
preset "jane" is `aai-jane`); your own voices are named by their UUID or slug.
Every voice speaks every `language`; see [Voices](/docs/tts/voices).

**language** `string` (default: english)

Spoken language: the language of the text the agent speaks, such as `spanish`.

**encoding** `string` (default: pcm\_s16le)

Output encoding. Only `pcm_s16le` (linear PCM) is decoded as raw audio by
LiveKit; `pcm_mulaw` / `pcm_alaw` are for telephony pipelines.

**sample\_rate** `int` (default: 24000)

Output sample rate in Hz.

**word\_timestamps** `boolean` (default: false)

Request `WordBoundaries` frames and expose them as an aligned transcript.

**api\_key** `string`

AssemblyAI API key. Falls back to the `ASSEMBLYAI_API_KEY` environment variable.

**base\_url** `string`

The WebSocket endpoint the plugin connects to, which decides where your text
and audio are processed. Left out, it is an endpoint of the US data zone. Set
`wss://tts.us.assemblyai.com/v1/ws` for the US or
`wss://tts.eu.assemblyai.com/v1/ws` for the EU.

To change the voice or language during a conversation, call
`tts.update_options(voice_id=..., language=...)`. It applies from the next reply,
which the plugin speaks on a new Streaming TTS session.

## Voices

Pass any of these to `assemblyai_tts.TTS(voice_id=...)` (or
`tts.update_options(voice_id=...)`) using the **Slug**. Every voice speaks every
`language`.

| Name | Language (Accent) | Gender | Description | Slug |
| --- | --- | --- | --- | --- |
| Titus | US (American) | Male | A young man with a deep, low-pitched voice speaking American English. | `aai-titus` |
| Ewald | US (American) | Male | A middle-aged man speaking American English at a fairly fast pace. | `aai-ewald` |
| Audrey | US (American) | Female | A middle-aged woman with an even, steady intonation speaks American English. | `aai-audrey` |
| Violet | US (American) | Female | A young woman with a balanced, natural, easy-to-listen-to voice speaking American English. | `aai-violet` |
| Troy | UK (Scottish) | Male | A young man with a soft-spoken, deep, low-pitched voice and expressive intonation of Scottish English. | `aai-troy` |
| Nash | UK (British) | Male | A middle-aged man speaking British English with a fairly even tone. | `aai-nash` |
| Nancy | UK (British) | Female | A middle-aged woman speaking British English with a fairly even tone. | `aai-nancy` |
| Carrie | UK (British) | Female | A middle-aged woman with a soft-spoken delivery in British English. | `aai-carrie` |
| Isadora | Portuguese | Female | A young woman speaking Portuguese with a fairly warm tone. | `aai-isadora` |
| Cloyd | English (UK) | Male | An older man with a high-pitched, fast-paced, energetic delivery in British English. | `aai-cloyd` |
| Kyla | English | Female | A young woman speaking American English at a fast pace. | `aai-kyla` |
| Steven | English (Australian) | Male | A middle-aged man speaking Australian English with a fairly even tone. | `aai-steven` |
| April | English | Female | A young woman speaking American English at a fast pace. | `aai-april` |
| Benton | English | Male | An older American man with an energetic delivery and expressive intonation, featuring a nasal quality. | `aai-benton` |
| Francisco | Spanish (Spain) | Male | A middle-aged man speaking Spanish from Spain with an energetic delivery and soft-spoken, warm, rich tone. | `aai-franciscoborja` |
| Eira | Spanish (Mexico) | Female | A young woman speaking Mexican Spanish with a high-pitched voice. | `aai-eira` |
| Donatien | French | Male | A middle-aged man with a high-pitched, soft-spoken voice and expressive intonation. | `aai-donatien` |
| Karin | German | Female | A middle-aged woman speaking German with energetic delivery, fast-paced and expressive intonation. | `aai-karin` |
| Amerigo | Italian | Male | An energetic, high-pitched, expressive Italian male voice with dynamic intonation. | `aai-amerigo` |

## Need help?

If you get stuck, contact our support team at
[support@assemblyai.com](mailto:support@assemblyai.com) or create a
[support ticket](https://www.assemblyai.com/contact/support). Include the `id`
from the `Begin` frame to help us look up your session.
