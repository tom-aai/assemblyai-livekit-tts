# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LiveKit TTS plugin for AssemblyAI's streaming (WebSocket) text-to-speech.

The service is WebSocket-only: a session opens a socket, the server replies with a
`Begin` frame echoing the configuration, and the client drives synthesis with
`Generate` (append text), `Flush` (speak everything buffered and mark a boundary),
and `Terminate` (flush the rest and close). Audio comes back as base64 `Audio`
frames, each `Flush` is answered by a `FlushDone`, and `Terminate` is answered by a
`Termination`. See the quickstart at https://assemblyai.com/docs/tts for details.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, replace
from urllib.parse import urlencode

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIError,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .log import logger
from .models import DEFAULT_LANGUAGE, DEFAULT_VOICE, TTSEncoding

# Global endpoint; routes to the nearest region. For pinned data residency use
# wss://streaming-tts.us.assemblyai.com/v1/ws or .eu. instead.
DEFAULT_BASE_URL = "wss://streaming-tts.usw2.assemblyai.com/v1/ws/"

NUM_CHANNELS = 1

# Mime type handed to the AudioEmitter. Only linear PCM is decoded as raw audio.
_MIME_TYPES: dict[str, str] = {
    "pcm_s16le": "audio/pcm",
    "pcm_mulaw": "audio/mulaw",
    "pcm_alaw": "audio/alaw",
}


@dataclass
class _TTSOptions:
    voice: str
    language: str
    encoding: TTSEncoding
    sample_rate: int
    word_timestamps: bool
    api_key: str
    base_url: str

    @property
    def mime_type(self) -> str:
        return _MIME_TYPES.get(self.encoding, "audio/pcm")

    def ws_url(self) -> str:
        params = {
            "voice": self.voice,
            "language": self.language,
            "sample_rate": self.sample_rate,
            "encoding": self.encoding,
            "word_boundaries": "true" if self.word_timestamps else "false",
        }
        return f"{self.base_url}?{urlencode(params)}"


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        voice: str = DEFAULT_VOICE,
        language: str = DEFAULT_LANGUAGE,
        encoding: TTSEncoding = "pcm_s16le",
        sample_rate: int = 24000,
        word_timestamps: bool = False,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
        text_pacing: tts.SentenceStreamPacer | bool = False,
    ) -> None:
        """Create a new AssemblyAI streaming TTS instance.

        Args:
            voice: Preset voice name, e.g. "jane". Voice and language must agree;
                see https://assemblyai.com/docs/tts/voices.
            language: Spoken language, e.g. "english".
            encoding: Output audio encoding. Defaults to "pcm_s16le" (linear PCM),
                which is the only encoding decoded as raw audio by LiveKit.
            sample_rate: Output sample rate in Hz. Defaults to 24000.
            word_timestamps: When True, request `WordBoundaries` frames and expose
                them as an aligned transcript. Off by default.
            api_key: AssemblyAI API key. Falls back to the ASSEMBLYAI_API_KEY
                environment variable.
            base_url: WebSocket endpoint. Defaults to the global streaming-tts host.
            http_session: An existing aiohttp session to reuse. One is created from
                the LiveKit HTTP context if omitted.
            tokenizer: Sentence tokenizer used to split streamed LLM text into
                `Generate`/`Flush` requests. Defaults to the blingfire tokenizer.
            text_pacing: Stream pacer for the TTS. Pass True for the default pacer,
                a SentenceStreamPacer instance, or False to disable.
        """
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamps,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )

        aai_api_key = api_key or os.environ.get("ASSEMBLYAI_API_KEY")
        if not aai_api_key:
            raise ValueError(
                "AssemblyAI API key is required, either as the `api_key` argument or "
                "the ASSEMBLYAI_API_KEY environment variable"
            )

        if encoding != "pcm_s16le":
            logger.warning(
                "encoding %s is not linear PCM; LiveKit decodes it best-effort and it "
                "is intended for telephony pipelines",
                encoding,
            )

        self._opts = _TTSOptions(
            voice=voice,
            language=language,
            encoding=encoding,
            sample_rate=sample_rate,
            word_timestamps=word_timestamps,
            api_key=aai_api_key,
            base_url=base_url,
        )
        self._session = http_session
        self._sentence_tokenizer = (
            tokenizer if is_given(tokenizer) else tokenize.blingfire.SentenceTokenizer()
        )
        self._stream_pacer: tts.SentenceStreamPacer | None = None
        if text_pacing is True:
            self._stream_pacer = tts.SentenceStreamPacer()
        elif isinstance(text_pacing, tts.SentenceStreamPacer):
            self._stream_pacer = text_pacing

    @property
    def model(self) -> str:
        return "streaming-tts"

    @property
    def provider(self) -> str:
        return "AssemblyAI"

    def update_options(
        self,
        *,
        voice: NotGivenOr[str] = NOT_GIVEN,
        language: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """Update the voice and/or language used for subsequent synthesis."""
        if is_given(voice):
            self._opts.voice = voice
        if is_given(language):
            self._opts.language = language

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _connect_ws(
        self, opts: _TTSOptions, timeout: float
    ) -> aiohttp.ClientWebSocketResponse:
        """Open a socket and consume the first frame (`Begin`, or `Error` on reject)."""
        session = self._ensure_session()
        try:
            # Pass the key raw; it is not a Bearer token.
            ws = await asyncio.wait_for(
                session.ws_connect(
                    opts.ws_url(), headers={"Authorization": opts.api_key}
                ),
                timeout,
            )
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except aiohttp.ClientResponseError as e:
            # Auth headers can appear in RequestInfo, so don't echo the exception.
            raise APIStatusError(
                message=e.message, status_code=e.status, request_id=None, body=None
            ) from None
        except Exception as e:
            # Transport errors can carry credentials in URLs.
            raise APIConnectionError(type(e).__name__) from None

        try:
            first = await ws.receive(timeout=timeout)
        except asyncio.TimeoutError:
            await ws.close()
            raise APITimeoutError() from None

        if first.type is not aiohttp.WSMsgType.TEXT:
            await ws.close()
            raise APIConnectionError(
                f"expected a Begin frame, got {first.type} (code {ws.close_code})"
            )

        data = json.loads(first.data)
        if data.get("type") == "Error":
            await ws.close()
            raise APIStatusError(
                message=data.get("error", "unknown error"),
                status_code=-1,
                request_id=None,
                body=data.get("error_code"),
            )

        logger.debug(
            "AssemblyAI TTS session established", extra={"configuration": data}
        )
        return ws

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> SynthesizeStream:
        return SynthesizeStream(tts=self, conn_options=conn_options)


def _parse_close(
    ws: aiohttp.ClientWebSocketResponse, request_id: str
) -> APIStatusError:
    # Close codes documented at https://assemblyai.com/docs/tts/error-handling
    # (e.g. 3008 session age limit, 3009 new-session rate, 3010 too much buffered).
    return APIStatusError(
        "AssemblyAI TTS connection closed unexpectedly",
        status_code=ws.close_code or -1,
        request_id=request_id,
        body=None,
    )


class ChunkedStream(tts.ChunkedStream):
    """Synthesize a fixed string over a one-shot WebSocket session."""

    def __init__(
        self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        ws = await self._tts._connect_ws(self._opts, self._conn_options.timeout)
        try:
            output_emitter.initialize(
                request_id=request_id,
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
                mime_type=self._opts.mime_type,
            )

            await ws.send_str(
                json.dumps({"type": "Generate", "text": self._input_text})
            )
            await ws.send_str(json.dumps({"type": "Terminate"}))

            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise _parse_close(ws, request_id)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                mtype = data.get("type")
                if mtype == "Audio":
                    output_emitter.push(base64.b64decode(data["audio"]))
                elif mtype == "Termination":
                    break
                elif mtype == "Error":
                    raise APIError(
                        f"AssemblyAI TTS error {data.get('error_code')}: {data.get('error')}"
                    )

            output_emitter.flush()
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await ws.close()


class SynthesizeStream(tts.SynthesizeStream):
    """Stream LLM text into synthesis, one `Generate`/`Flush` per sentence."""

    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        request_id = utils.shortuuid()
        segment_id = utils.shortuuid()
        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
            mime_type=self._opts.mime_type,
            stream=True,
        )

        sent_tokenizer_stream = self._tts._sentence_tokenizer.stream()
        if self._tts._stream_pacer:
            sent_tokenizer_stream = self._tts._stream_pacer.wrap(
                sent_stream=sent_tokenizer_stream,
                audio_emitter=output_emitter,
            )

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_tokenizer_stream.flush()
                    continue
                sent_tokenizer_stream.push_text(data)
            sent_tokenizer_stream.end_input()

        async def _send_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            async for ev in sent_tokenizer_stream:
                self._mark_started()
                # Consecutive Generate texts are joined verbatim, so keep a trailing
                # space between sentences. Flush speaks what is buffered so far.
                await ws.send_str(
                    json.dumps({"type": "Generate", "text": ev.token + " "})
                )
                await ws.send_str(json.dumps({"type": "Flush"}))
            # Terminate flushes anything still owed, then the server replies Termination.
            await ws.send_str(json.dumps({"type": "Terminate"}))

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse) -> None:
            segment_started = False
            # Word-boundary times are relative to each flush; accumulate the audio
            # duration of completed flushes to place them on the session timeline.
            flush_base_ms = 0.0
            next_base_ms = 0.0
            while True:
                msg = await ws.receive(timeout=self._conn_options.timeout)
                if msg.type in (
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                ):
                    raise _parse_close(ws, request_id)
                if msg.type is not aiohttp.WSMsgType.TEXT:
                    continue

                data = json.loads(msg.data)
                mtype = data.get("type")
                if mtype == "Audio":
                    if not segment_started:
                        output_emitter.start_segment(segment_id=segment_id)
                        segment_started = True
                    output_emitter.push(base64.b64decode(data["audio"]))
                elif mtype == "FlushDone":
                    flush_base_ms = next_base_ms
                    next_base_ms += float(data.get("audio_duration_ms", 0.0))
                elif mtype == "WordBoundaries":
                    if self._opts.word_timestamps:
                        _emit_word_boundaries(output_emitter, data, flush_base_ms)
                elif mtype == "Termination":
                    break
                elif mtype == "Error":
                    raise APIError(
                        f"AssemblyAI TTS error {data.get('error_code')}: {data.get('error')}"
                    )

            if segment_started:
                output_emitter.end_input()

        ws = await self._tts._connect_ws(self._opts, self._conn_options.timeout)
        try:
            tasks = [
                asyncio.create_task(_input_task()),
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
            ]
            try:
                await asyncio.gather(*tasks)
            finally:
                await sent_tokenizer_stream.aclose()
                await utils.aio.gracefully_cancel(*tasks)
        except asyncio.TimeoutError:
            raise APITimeoutError() from None
        except APIError:
            raise
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            await ws.close()


def _emit_word_boundaries(
    output_emitter: tts.AudioEmitter, data: dict, base_ms: float
) -> None:
    for word in data.get("words", []):
        text = word.get("word")
        if text is None or "start" not in word or "end" not in word:
            continue
        output_emitter.push_timed_transcript(
            TimedString(
                text=text + " ",
                start_time=(base_ms + float(word["start"])) / 1000.0,
                end_time=(base_ms + float(word["end"])) / 1000.0,
            )
        )
