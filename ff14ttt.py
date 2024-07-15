from gradio_client import Client
import asyncio
import websockets
import json
import playsound as ps
from logging import getLogger, StreamHandler, DEBUG
import sys
import winsound
import pyperclip
import os

# HuggingFace interface API
access_token = os.environ.get('HF_TOKEN')
client = Client("RafaG/TTS-Rapido", hf_token=access_token)

logger = getLogger(__name__)
logger.addHandler(StreamHandler(stream=sys.stdout))
logger.setLevel(DEBUG)

# Settings for the warning beep
frequency = 500
duration = 50

async def connect_to_websocket(uri):
    while True:
        try:
            async with websockets.connect(uri, ping_interval=10, ping_timeout=5) as websocket:
                logger.info("Connected to WebSocket server")
                while True:
                    try:
                        message = await websocket.recv()
                        logger.info(f"Received message: {message}")
                        await process_message(message)
                    except websockets.ConnectionClosedError as e:
                        logger.warning(f"WebSocket connection closed while receiving message: {e}")
                        winsound.Beep(frequency, duration)
                        break  # Break inner loop to reconnect
                    except Exception as e:
                        logger.error(f"Error receiving message: {e}")
                        break  # Break inner loop to reconnect
        except (websockets.InvalidStatusCode, websockets.ConnectionClosedError, Exception) as e:
            logger.error(f"WebSocket connection failed: {e}")

        logger.info(f"Reconnecting...")

async def process_message(message):
    try:
        message_data = json.loads(message)
        payload = message_data.get("Payload")

        if not payload:
            logger.warning("Received message without 'Payload'")
            return

        pyperclip.copy(payload)
        # Use the HF client to read the payload (text), in japanese (ja-JP-NanamiNeural), at 90% speed (-10), with silence cutter (True)
        result = client.predict(
            payload,
            "ja-JP-NanamiNeural",
            -10,
            True,
            api_name="/controlador_generate_audio"
        )

        # Play the text to speech
        if result and len(result) > 1:
            ps.playsound(result)
        else:
            logger.warning("TTS result is empty or malformed")

    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON message: {e}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

async def main(uri):
    tasks = [
        connect_to_websocket(uri),
    ]

    await asyncio.gather(*tasks)

if __name__ == "__main__":
    uri = "ws://127.0.0.1:3000/Messages"
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main(uri))
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    finally:
        loop.close()
