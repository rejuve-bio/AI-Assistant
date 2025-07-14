import logging
import os
from pyt2s.services import stream_elements

logger = logging.getLogger(__name__)


class TTSManager:
    # Text-to-Speech manager for generating audio files using pyt2s
    def __init__(self):
        # Initialize TTS manager
        self.available_voices = {
            "default": stream_elements.Voice.Russell.value,
            "russell": stream_elements.Voice.Russell.value,
        }

    def generate_audio(self, text, audio_path, max_length=5000, voice="default"):
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(audio_path), exist_ok=True)

            # Limit text length to avoid very long audio files
            if len(text) > max_length:
                text = text[:max_length] + "..."

            # Get the voice parameter
            voice_param = self.available_voices.get(
                voice.lower(), stream_elements.Voice.Russell.value
            )

            # Request TTS from stream_elements
            logger.info(f"Generating audio with voice: {voice}")
            data = stream_elements.requestTTS(text, voice_param)

            # Save the audio data to file
            with open(audio_path, "wb") as file:
                file.write(data)

            logger.info(f"Audio generated successfully: {audio_path}")
            return True

        except Exception as e:
            logger.error(f"Error generating audio: {e}")
            return False

    def generate_summary_audio(self, text, user_id, pdf_id, voice="default"):
        audio_dir = os.path.join("storage", "audio", str(user_id), "summaries")
        audio_path = os.path.join(audio_dir, f"{pdf_id}.mp3")
        return self.generate_audio(text, audio_path, voice=voice)

    def generate_query_audio(self, text, user_id, query_id, voice="default"):
        audio_dir = os.path.join("storage", "audio", str(user_id), "queries")
        audio_path = os.path.join(audio_dir, f"{query_id}.mp3")
        return self.generate_audio(text, audio_path, voice=voice)


tts_manager = TTSManager()
