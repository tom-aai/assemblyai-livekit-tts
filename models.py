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

from typing import Literal

# AssemblyAI streaming TTS output encodings. Only linear 16-bit PCM is handled as
# raw audio by the LiveKit AudioEmitter; mu-law/a-law are listed for completeness
# but are intended for telephony pipelines that expect those codecs.
TTSEncoding = Literal["pcm_s16le", "pcm_mulaw", "pcm_alaw"]

# The default preset voice from the quickstart. See the full catalog at
# https://assemblyai.com/docs/tts/voices (voice and language must agree).
DEFAULT_VOICE = "jane"
DEFAULT_LANGUAGE = "english"
