import logging
from pyt2s.services import stream_elements

logger = logging.getLogger(__name__)


class TTSManager:
    # Text-to-Speech manager for generating audio data using pyt2s
    def __init__(self):
        self.available_voices = {
            "default": stream_elements.Voice.Russell.value,
            "russell": stream_elements.Voice.Russell.value,
        }

    def generate_audio_data(self, text, max_length=5000, voice="default"):
        # Generate audio data in memory without saving to file
        try:
            if len(text) > max_length:
                text = text[:max_length] + "..."
            voice_param = self.available_voices.get(
                voice.lower(), stream_elements.Voice.Russell.value
            )
            logger.info(f"Generating audio data with voice: {voice}")
            data = stream_elements.requestTTS(text, voice_param)
            logger.info(f"Audio data generated successfully: {len(data)} bytes")
            return data
        except Exception as e:
            logger.error(f"Error generating audio data: {e}")
            return None

    def generate_audio_on_demand(self, text, voice="default"):
        # Generate audio on-demand and return the data
        return self.generate_audio_data(text, voice=voice)


tts_manager = TTSManager()
