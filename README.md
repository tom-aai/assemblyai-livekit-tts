> ## Documentation Index
> Fetch the complete documentation index at: https://assemblyai.com/docs/llms.txt
> Use this file to discover all available pages before exploring further.

# Quickstart

> Open a WebSocket, send a line of text, and write the streamed audio to a WAV file.

The Streaming TTS API turns text into speech over a WebSocket, sending audio back in chunks as it is synthesized rather than making you wait for a finished file. This page gets you from an empty file to playable audio.

> [!NOTE]
>
> This page uses the WebSocket API, which suits text that arrives a piece at a
> time and conversations held on one connection. To turn a whole text into
> speech with one request, use the [HTTP API](/docs/tts/http-api) instead: it
> streams the audio as the response body, takes a single `curl` command, and
> also answers the [OpenAI speech API shape](/docs/tts/openai-compatibility). SDK
> support for the WebSocket API is a work in progress, so every example here
> opens a socket with a WebSocket library directly.

## Before you begin

* **An API key.** Copy it from [API Keys](https://www.assemblyai.com/dashboard/home) and set it once:

  ```bash theme={null}
  export ASSEMBLYAI_API_KEY=<your-key>
  ```

* **A WebSocket client.** The example uses `additional_headers`, which needs `websockets` 14 or newer:

  ```bash theme={null}
  pip install -U websockets
  ```

## Generate your first audio

Connect and read the `Begin` frame, send the text, then send a `Flush` to conclude the request. Audio arrives as `Audio` frames, and a `FlushDone` tells you that all the audio for the request has been delivered.

```python theme={null}
import asyncio
import base64
import json
import os
import struct

import websockets

URL = (
    "wss://tts.us.assemblyai.com/v1/ws"
    "?voice_id=aai-jane&language=english"
    "&sample_rate=24000&encoding=pcm_s16le"
)


def wav_header(n_bytes, rate, channels=1, bits=16):
    byte_rate = rate * channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36 + n_bytes) + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate,
                      channels * bits // 8, bits)
        + b"data" + struct.pack("<I", n_bytes)
    )


async def main():
    # The raw key works; a "Bearer " prefix is accepted too.
    headers = {"Authorization": os.environ["ASSEMBLYAI_API_KEY"]}

    async with websockets.connect(URL, additional_headers=headers) as ws:
        # The server sends the first frame. Begin carries the session
        # configuration; a rejected key or parameter arrives as Error instead.
        first = json.loads(await ws.recv())
        if first["type"] == "Error":
            raise SystemExit(f"{first['error_code']}: {first['error']}")
        config = first["configuration"]
        print(f"voice {config['voice_id']}, model {config['model']}, "
              f"{config['sample_rate']} Hz {config['encoding']}")

        await ws.send(json.dumps({
            "type": "Generate",
            "text": "Hi, thanks for calling. How can I help you today?",
        }))
        await ws.send(json.dumps({"type": "Flush"}))

        pcm = bytearray()
        async for raw in ws:
            message = json.loads(raw)
            kind = message.get("type")

            if kind == "Audio":
                pcm += base64.b64decode(message["audio"])
            elif kind == "FlushDone":
                print(f"flush {message['flush_id']}: "
                      f"{message['audio_duration_ms']} ms of audio")
                await ws.send(json.dumps({"type": "Terminate"}))
            elif kind == "Termination":
                break
            elif kind == "Error":
                raise SystemExit(f"{message['error_code']}: {message['error']}")
            # Other frame types, such as WordBoundaries, are ignored here.

    with open("out.wav", "wb") as f:
        f.write(wav_header(len(pcm), config["sample_rate"]))
        f.write(pcm)

    print(f"wrote out.wav ({len(pcm)} bytes of PCM)")


asyncio.run(main())
```

Run it and you'll get a playable `out.wav`. The output looks like this:

```text theme={null}
voice aai-jane, model assemblyai-tts-1, 24000 Hz pcm_s16le
flush 0: 3120 ms of audio
wrote out.wav (149760 bytes of PCM)
```

Four things about this run:

* **The server speaks first.** `Begin` echoes the configuration the session actually got, which is why the program takes the WAV sample rate from it rather than from its own request. It holds every connection parameter except `token` and `model_id` with its effective value, plus the model, the session's limits and the names of any query parameters the server ignored; [Read the configuration back](/docs/tts/connect-parameters#read-the-configuration-back) describes these keys. The object may gain keys over time, so read the ones you need rather than checking it against a fixed set.
* **Errors arrive on an open socket.** When a credential or a connect parameter is rejected, an unknown `voice_id` included, an `Error` frame arrives in place of `Begin` and the socket closes, which is why the program reads one frame before it sends anything. See [Error handling](/docs/tts/error-handling).
* **The duration varies between runs.** Synthesis is not deterministic, so the same sentence can come back a few hundred milliseconds longer or shorter. What does not vary: at 24 kHz 16-bit mono, the byte count is `audio_duration_ms` x 48, give or take the rounding to whole milliseconds.
* **A frame you did not see.** `word_boundaries` is on by default, so a `WordBoundaries` frame normally follows the `FlushDone` with one entry per word (`word`, `start` and `end` in milliseconds, and a `confidence`). The loop above ignores it, and you should ignore any frame type you do not handle. See [Word boundaries](/docs/tts/word-boundaries).

If the socket closes without an `Error` frame, for example on a network drop, the `async for` loop raises `websockets.exceptions.ConnectionClosedError`. [Error handling](/docs/tts/error-handling) lists the close codes you can see either way.

The URL above is the US endpoint, which processes your text and audio in the US. To keep processing in the EU, connect to `wss://tts.eu.assemblyai.com/v1/ws` instead. See [Cloud endpoints and data residency](/docs/tts/endpoints-and-data-zones).

## Flush marks the end of a request

`Generate` appends text to the session buffer. Short text is not spoken until you flush, and the socket stays open and healthy while it waits, until two minutes pass with no frame in either direction and the [idle timeout](/docs/tts/connect-parameters#inactivity-timeout) closes it with code `3006`. That makes a missing `Flush` look like a hang rather than a mistake.

Longer text does not wait. Once the buffer holds enough complete sentences, the server starts speaking them before any `Flush`, under the current `flush_id`, and text streamed over several `Generate` frames, as an LLM produces it, starts once its first sentences are complete. See [Generate](/docs/tts/message-frames#generate). Only `Flush` (or `Terminate`, which flushes for you) guarantees that everything you sent is spoken and gives you the `FlushDone` boundary:

```text theme={null}
short text:  Generate {text}  ->  buffered, silent  ->  Flush  ->  Audio {audio, flush_id} ...  ->  FlushDone {flush_id, audio_duration_ms}
long text:   Generate {text}  ->  Audio {audio, flush_id} ...  ->  Flush  ->  remaining Audio ...  ->  FlushDone {flush_id, audio_duration_ms}
```

Send `Flush` at every point where you need everything sent so far to be spoken and marked with a `FlushDone`. If you are streaming text out of an LLM, that usually means once per sentence or clause, not once per token. Two details matter when you send text in pieces:

* Consecutive `Generate` texts are joined exactly as sent, with no separator. `Good` followed by `morning.` is spoken as `Goodmorning.`, so keep the space at the edge of each piece.
* `flush_id` is the request counter. It starts at `0`, each request you conclude with `Flush` takes the next id, and every `Audio` frame carries the `flush_id` of the request it belongs to. After a `Cancel`, which interrupts a request, the next id can skip numbers, so read ids from the frames rather than counting them yourself.

## Several requests in one session

A session is not a single request. Keep one socket open and send as many `Generate` and `Flush` pairs as the conversation needs. Opening a socket per sentence pays the connection setup every time and quickly hits the per-account limit on new sessions per minute, which closes the connection with code `3009`. Two rules keep a multi-request session in order:

* **Every `Flush` gets exactly one `FlushDone`, in order.** Count the flushes you send and the `FlushDone` frames you receive, and send `Terminate` when they match. `Terminate` also flushes any text you have not concluded yet and delivers the `FlushDone` frames still owed before `Termination`, so sending it early does not lose audio, but the count is what tells you the session is idle.
* **Keep a bounded number of requests in flight.** The server works flushes one after another and accepts text faster than it speaks it, but not without bound: a long document sent as one burst closes the session with code `3010`. Send the next sentence when a `FlushDone` arrives rather than sending everything at once. See [Limits](/docs/tts/limits).

```python theme={null}
# Fragment: a replacement for main() in the program above. The imports,
# URL, wav_header() and the asyncio.run(main()) line stay the same.
async def main():
    headers = {"Authorization": os.environ["ASSEMBLYAI_API_KEY"]}
    sentences = [  # your sentence splitter goes here; each sentence is its own request
        "Hi, thanks for calling.",
        "You have reached the accounts team.",
        "I can see your account details now.",
        "There is one open invoice from last month.",
        "Would you like to pay it today?",
    ]
    MAX_IN_FLIGHT = 3  # flushes sent but not yet answered by FlushDone

    async with websockets.connect(URL, additional_headers=headers) as ws:
        first = json.loads(await ws.recv())
        if first["type"] == "Error":
            raise SystemExit(f"{first['error_code']}: {first['error']}")
        config = first["configuration"]

        async def speak(text):
            await ws.send(json.dumps({"type": "Generate", "text": text}))
            await ws.send(json.dumps({"type": "Flush"}))

        queue = list(sentences)
        in_flight = 0
        while queue and in_flight < MAX_IN_FLIGHT:
            await speak(queue.pop(0))
            in_flight += 1

        pcm = bytearray()
        async for raw in ws:
            message = json.loads(raw)
            kind = message.get("type")

            if kind == "Audio":
                pcm += base64.b64decode(message["audio"])
            elif kind == "FlushDone":
                print(f"flush {message['flush_id']}: "
                      f"{message['audio_duration_ms']} ms of audio")
                in_flight -= 1
                while queue and in_flight < MAX_IN_FLIGHT:
                    await speak(queue.pop(0))
                    in_flight += 1
                if not queue and in_flight == 0:
                    await ws.send(json.dumps({"type": "Terminate"}))
            elif kind == "Termination":
                break
            elif kind == "Error":
                raise SystemExit(f"{message['error_code']}: {message['error']}")

    with open("out.wav", "wb") as f:
        f.write(wav_header(len(pcm), config["sample_rate"]))
        f.write(pcm)

    print(f"wrote out.wav ({len(pcm)} bytes of PCM)")
```

## Audio has no header

`Audio.audio` is base64-encoded raw codec bytes. There is no WAV header on it, so it will not play if you hand it straight to an `<audio>` element or a `Blob`. You either add a header yourself, as the example above does for the file on disk, or decode the samples and play them directly. The 16-bit PCM header in the example is right for `pcm_s16le` only; `pcm_mulaw` and `pcm_alaw` need a different format code. See [Handling audio output](/docs/tts/audio-output).

## Integrations

Streaming TTS has no official SDK plugins yet, so framework integrations are
maintained separately while they stabilize. The sections below walk through
wiring Streaming TTS into other voice-AI frameworks.

### LiveKit (Beta)

> [!NOTE]
>
> **Internal / Beta.** This integration uses an in-house AssemblyAI TTS plugin
> that is **not yet published** to the official
> [`livekit-plugins`](https://github.com/livekit/agents) repository. You can't
> `uv add` it from PyPI. You clone the plugin repo and install it into your
> agent, as described below. APIs may change before the public release.

This guide covers integrating AssemblyAI's **Streaming TTS** into a LiveKit
voice agent using the [Agents framework](https://docs.livekit.io/agents/) (1.8+).
LiveKit Agents is an open-source framework for building real-time voice agents:
your agent joins a LiveKit room as a participant and exchanges audio with users
over WebRTC or SIP. An `AgentSession` wires together speech-to-text, an LLM, and
text-to-speech, and handles orchestration (turn detection, interruptions).

#### Requirements

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

Clone the plugin repo inside your agent project and install it into the project's
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
> Streaming TTS plugin doesn't support that markup, so leaving expressive mode
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

> The older `voice` argument (a bare name like `jane`) is deprecated; it maps to
> the matching `voice_id` but is refused by the service, so pass `voice_id`.

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
`wss://tts.eu.assemblyai.com/v1/ws` for the EU; see
[Cloud endpoints and data residency](/docs/tts/endpoints-and-data-zones).

To change the voice or language during a conversation, call
`tts.update_options(voice_id=..., language=...)`. It applies from the next reply,
which the plugin speaks on a new Streaming TTS session.

## Related

* [Stream to a browser](/docs/tts/getting-started/stream-to-a-browser): play the audio as it arrives, instead of waiting for the whole thing.
* [Voices](/docs/tts/voices): AssemblyAI's voices and the languages they speak.
* [Connect parameters](/docs/tts/connect-parameters): sample rate, encoding, word boundaries, and inactivity timeout.
* [Message frames](/docs/tts/message-frames): every frame in the protocol, in both directions.
* [Limits](/docs/tts/limits): text, session, and rate limits, and how each one reports a breach.
* [HTTP API](/docs/tts/http-api): the same speech from one HTTP request, with `curl`.
* [Cloud endpoints and data residency](/docs/tts/endpoints-and-data-zones): the US and EU hosts.

## Need help?

If you get stuck, contact our support team at
[support@assemblyai.com](mailto:support@assemblyai.com) or create a
[support ticket](https://www.assemblyai.com/contact/support). Include the `id`
from the `Begin` frame to help us look up your session.
