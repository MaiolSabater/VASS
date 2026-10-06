"""BLE LED strip control for VASS (HappyLighting / QHM controller).

bleak is async while VASS is synchronous, so the BLE client lives on a private
asyncio loop in a daemon thread; the public functions block until the BLE
operation finishes and return a short sentence for Vass to speak.

The connection is kept open between commands (the strip accepts only one
client, so the HappyLighting app must be closed). If it drops, the next
command reconnects automatically.
"""

import asyncio
import atexit
import threading

from bleak import BleakClient
from bleak.exc import BleakError

import config
from light import CMD_POWER_OFF, CMD_POWER_ON, color_cmd

# Spoken color name -> RGB. Multi-word names are matched before single words.
COLORS = {
    "warm white": (255, 140, 40),
    "light blue": (80, 160, 255),
    "dark blue": (0, 0, 140),
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "yellow": (255, 180, 0),
    "orange": (255, 60, 0),
    "purple": (140, 0, 255),
    "violet": (140, 0, 255),
    "pink": (255, 40, 120),
    "magenta": (255, 0, 255),
    "cyan": (0, 255, 255),
    "turquoise": (0, 255, 140),
}

_BLE_ERRORS = (BleakError, asyncio.TimeoutError, OSError)

_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True, name="lights-ble").start()
_client: BleakClient | None = None


def _run(coro, timeout: float):
    return asyncio.run_coroutine_threadsafe(coro, _loop).result(timeout)


def is_connected() -> bool:
    return _client is not None and _client.is_connected


async def _connect() -> None:
    global _client
    if is_connected():
        return
    client = BleakClient(config.LIGHTS_ADDRESS, timeout=config.LIGHTS_CONNECT_TIMEOUT)
    await client.connect()
    _client = client


async def _write(data: bytes, response: bool) -> None:
    await _connect()
    await _client.write_gatt_char(config.LIGHTS_WRITE_CHAR, data, response=response)


async def _send(*commands: tuple[bytes, bool]) -> None:
    """Write commands in order; on a BLE error reconnect once and retry the whole batch."""
    global _client
    for attempt in range(2):
        try:
            for data, response in commands:
                await _write(data, response)
            return
        except _BLE_ERRORS:
            if attempt:
                raise
            _client = None  # stale link: force a fresh connect


def _do(coro, ok_reply: str) -> str:
    try:
        _run(coro, timeout=config.LIGHTS_CONNECT_TIMEOUT + 10)
        return ok_reply
    except _BLE_ERRORS as e:
        print(f"[Lights] {e!r}")
        return "Couldn't reach the lights. Is the app still connected to them?"


def connect() -> str:
    try:
        _run(_connect(), timeout=config.LIGHTS_CONNECT_TIMEOUT + 10)
        return "Connected to lights."
    except _BLE_ERRORS as e:
        print(f"[Lights] connect failed: {e!r}")
        return "Connection to the lights failed."


def disconnect() -> str:
    global _client
    if is_connected():
        try:
            _run(_client.disconnect(), timeout=10)
        except _BLE_ERRORS:
            pass
    _client = None
    return "Disconnected from lights."


def set_color(name: str) -> str:
    # The controller ignores colors while powered off, so power on first.
    return _do(_send((CMD_POWER_ON, True), (color_cmd(*COLORS[name]), False)),
               f"Lights set to {name}.")


def power_on() -> str:
    return _do(_send((CMD_POWER_ON, True)), "Lights on.")


def power_off() -> str:
    # This controller's "off" is a dim blue standby glow (hardware), not fully dark.
    return _do(_send((CMD_POWER_OFF, True)), "Lights off.")


atexit.register(lambda: is_connected() and disconnect())
