"""Standalone test script for a HappyLighting (com.xiaoyu.hlight) BLE RGB LED strip.

Usage:
    python light.py scan                 # list nearby BLE devices
    python light.py services             # list GATT services/characteristics of DEVICE_ADDRESS
    python light.py test                 # cycle red -> green -> blue -> white -> off
    python light.py color 255 80 0       # set a single color
    python light.py color 255 80 0 -b 30 # same color at 30% brightness
    python light.py on | off             # power on / off
    python light.py raw CC 24 33         # send arbitrary hex bytes (for trying command variants)
    python light.py probe-off            # try known off-command variants one by one, watch the strip
    python light.py query                # ask the strip for its state (EF 01 77) and print the reply

Optional: --address XX:XX:... and --char <uuid> override the constants below.
"""

import argparse
import asyncio
import sys

from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# ============================================================================
# FILL THESE IN after running `python light.py scan` / `python light.py services`
# ============================================================================
DEVICE_ADDRESS = "69:AB:00:CE:5C:8A"  # QHM-5C8A
WRITE_CHAR_UUID = "0000ffd9-0000-1000-8000-00805f9b34fb"  # confirmed on QHM-5C8A (service ffd5)
# ============================================================================

SCAN_TIMEOUT = 8.0
STEP_DELAY = 1.0

# HappyLighting power commands. QHM-5C8A only obeys these as ACKNOWLEDGED writes
# (write-with-response); sent without response they are silently ignored.
# While off, the controller shows a blue standby glow and ignores color commands.
CMD_POWER_ON = bytes([0xCC, 0x23, 0x33])
CMD_POWER_OFF = bytes([0xCC, 0x24, 0x33])


def color_cmd(r, g, b, brightness=100):
    # RGB controllers have no separate brightness setting: dimming = scaling the channels.
    r, g, b = (round(v * brightness / 100) for v in (r, g, b))
    return bytes([0x56, r & 0xFF, g & 0xFF, b & 0xFF, 0x00, 0xF0, 0xAA])



def print_single_connection_reminder():
    print("!" * 70)
    print("! The strip accepts only ONE BLE connection at a time.")
    print("! Fully close the HappyLighting app (swipe it away / force stop) and")
    print("! turn off phone Bluetooth if needed, or this script won't find/connect.")
    print("!" * 70)


async def scan():
    print_single_connection_reminder()
    print(f"Scanning for {SCAN_TIMEOUT:.0f}s...\n")
    found = await BleakScanner.discover(timeout=SCAN_TIMEOUT, return_adv=True)
    rows = sorted(found.values(), key=lambda da: da[1].rssi, reverse=True)
    print(f"{'ADDRESS':<20} {'RSSI':>5}  NAME / ADVERTISED DATA")
    for device, adv in rows:
        name = adv.local_name or device.name or "(no name)"
        print(f"{device.address:<20} {adv.rssi:>5}  {name}")
        # Nameless devices are still identifiable by what they advertise:
        # LED controllers typically list a short vendor service like ffe0 / ffd5 / fff0.
        for uuid in adv.service_uuids:
            short = uuid[4:8] if uuid.endswith("-0000-1000-8000-00805f9b34fb") else uuid
            print(f"{'':<28}service  {short}")
        for company_id, data in adv.manufacturer_data.items():
            print(f"{'':<28}mfr 0x{company_id:04X}  {data.hex(' ').upper()}")
    print(f"\n{len(rows)} device(s). Look for names like LEDBLE-*, QHM-*, Triones*, "
          "ELK-*, or a strong RSSI near your strip. Unplug the strip and rescan "
          "if unsure: the entry that disappears is yours.")


def list_services(client):
    print("\nGATT services / characteristics:")
    for service in client.services:
        print(f"[Service] {service.uuid}  {service.description}")
        for char in service.characteristics:
            props = ",".join(char.properties)
            print(f"    [Char] {char.uuid}  ({props})  {char.description}")


def pick_write_char(client, preferred_uuid):
    """Return the characteristic to write to: preferred UUID if present, else first writable one."""
    writable = [
        c for s in client.services for c in s.characteristics
        if "write" in c.properties or "write-without-response" in c.properties
    ]
    if preferred_uuid:
        for c in writable:
            if c.uuid.lower() == preferred_uuid.lower():
                return c
        print(f"Preferred characteristic {preferred_uuid} not found or not writable; falling back.")
    if not writable:
        raise BleakError("No writable characteristic found on this device.")
    # Prefer vendor (non-standard) characteristics such as ffe1/ffd9 over generic ones.
    vendor = [c for c in writable if not c.uuid.startswith("00002a")]
    chosen = (vendor or writable)[0]
    if len(writable) > 1:
        print("Multiple writable characteristics: " + ", ".join(c.uuid for c in writable))
    return chosen


async def write(client, char, data, label, response=None):
    # These strips usually expose write-without-response; use acknowledged writes only if that's all there is.
    if response is None:
        response = "write-without-response" not in char.properties
    print(f"-> {label:<10} {data.hex(' ').upper()}")
    await client.write_gatt_char(char, data, response=response)


# Off-command variants from the HappyLighting/Triones, Magic Home and ELK-BLEDOM families,
# looking for a fully dark off (CC 24 33 acknowledged = blue standby, already known).
# Each entry: (label, bytes, characteristic UUID or None for the main one)
FFD1 = "0000ffd1-0000-1000-8000-00805f9b34fb"
OFF_CANDIDATES = [
    ("black", color_cmd(0, 0, 0), None),
    ("warm-white mode, level 0", bytes.fromhex("56 00 00 00 00 0F AA"), None),
    ("RGB+W mode, all zero", bytes.fromhex("56 00 00 00 00 FF AA"), None),
    ("Magic Home off", bytes.fromhex("71 24 0F A4"), None),
    ("ELK-BLEDOM off", bytes.fromhex("7E 00 04 00 00 00 FF 00 EF"), None),
    ("ELK-BLEDOM off (alt)", bytes.fromhex("7E 04 04 00 00 00 FF 00 EF"), None),
    ("CC 24 33 on ffd1", bytes.fromhex("CC 24 33"), FFD1),
    ("Magic Home off on ffd1", bytes.fromhex("71 24 0F A4"), FFD1),
    ("near-black 01 01 01", color_cmd(1, 1, 1), None),
]
PROBE_SHOW = 1.5   # seconds of red before each candidate
PROBE_WATCH = 3.0  # seconds to watch after each candidate


async def probe_off(client, char):
    red = color_cmd(255, 0, 0)
    print("Watch the strip. Before every candidate it is powered on and set to red;\n"
          "note the LABEL (e.g. #4b) after which it goes fully dark.\n")
    for i, (label, data, uuid) in enumerate(OFF_CANDIDATES, 1):
        target = client.services.get_characteristic(uuid) if uuid else char
        if target is None:
            print(f"#{i} {label}: characteristic {uuid} not present, skipped")
            continue
        # a = unacknowledged, b = acknowledged (only if the characteristic supports it)
        modes = [("a", False)] + ([("b", True)] if "write" in target.properties else [])
        for suffix, response in modes:
            await write(client, char, CMD_POWER_ON, "power on", response=True)
            await write(client, char, red, "red")
            await asyncio.sleep(PROBE_SHOW)
            tag = f"#{i}{suffix}"
            print(f"{tag} {label} ({'acknowledged' if response else 'unacknowledged'})")
            try:
                await write(client, target, data, tag, response=response)
            except BleakError as e:
                print(f"   write failed: {e!r}")
            await asyncio.sleep(PROBE_WATCH)
    await write(client, char, CMD_POWER_ON, "power on", response=True)
    await write(client, char, red, "red")
    print("\nProbe finished.")


async def query_state(client, char):
    replies = []

    def on_notify(sender, data):
        replies.append(data)
        print(f"<- {sender.uuid[4:8]}  {bytes(data).hex(' ').upper()}")

    notifiers = [c for s in client.services for c in s.characteristics if "notify" in c.properties]
    for c in notifiers:
        await client.start_notify(c, on_notify)
    await write(client, char, bytes.fromhex("EF0177"), "query")
    await asyncio.sleep(2.0)
    for c in notifiers:
        await client.stop_notify(c)
    if not replies:
        print("No reply: this firmware doesn't answer the EF 01 77 status query.")


async def run(address, char_uuid, action, payload=None):
    if not address:
        sys.exit("No address. Set DEVICE_ADDRESS at the top of light.py or pass --address.")
    print_single_connection_reminder()
    print(f"Connecting to {address}...")
    try:
        async with BleakClient(address, timeout=20.0) as client:
            print(f"Connected: {client.is_connected}")
            list_services(client)
            if action == "services":
                return
            char = pick_write_char(client, char_uuid)
            print(f"\nUsing write characteristic: {char.uuid} ({','.join(char.properties)})\n")

            if action == "test":
                await write(client, char, CMD_POWER_ON, "power on", response=True)
                await asyncio.sleep(STEP_DELAY)
                for label, (r, g, b) in [("red", (255, 0, 0)), ("green", (0, 255, 0)),
                                         ("blue", (0, 0, 255)), ("white", (255, 255, 255))]:
                    await write(client, char, color_cmd(r, g, b), label)
                    await asyncio.sleep(STEP_DELAY)
                await write(client, char, CMD_POWER_OFF, "power off", response=True)
            elif action == "color":
                await write(client, char, payload, "color")
            elif action == "raw":
                await write(client, char, payload, "raw")
            elif action == "on":
                await write(client, char, CMD_POWER_ON, "power on", response=True)
            elif action == "off":
                await write(client, char, CMD_POWER_OFF, "power off", response=True)
            elif action == "probe-off":
                await probe_off(client, char)
            elif action == "query":
                await query_state(client, char)
            await asyncio.sleep(0.5)  # let the last write flush before disconnecting
            print("\nDone.")
    except (BleakError, asyncio.TimeoutError, OSError) as e:
        print(f"\nBLE error: {e!r}")
        print("Is the HappyLighting app fully closed? Is the strip powered and in range? "
              "Is the address correct (run `python light.py scan`)?")
        sys.exit(1)


def main():
    p = argparse.ArgumentParser(description="HappyLighting BLE LED strip test")
    p.add_argument("action", choices=["scan", "services", "test", "color", "on", "off", "raw", "probe-off", "query"])
    p.add_argument("values", nargs="*", help="R G B (0-255) for 'color'; hex bytes for 'raw'")
    p.add_argument("--address", default=DEVICE_ADDRESS)
    p.add_argument("--char", default=WRITE_CHAR_UUID, help="write characteristic UUID")
    p.add_argument("-b", "--brightness", type=int, default=100, help="0-100 %% for 'color'")
    args = p.parse_args()

    if args.action == "scan":
        asyncio.run(scan())
        return
    payload = None
    if args.action == "color":
        try:
            rgb = [int(v) for v in args.values]
        except ValueError:
            rgb = []
        if len(rgb) != 3 or not all(0 <= v <= 255 for v in rgb):
            p.error("color needs three values 0-255, e.g. `color 255 80 0`")
        if not 0 <= args.brightness <= 100:
            p.error("--brightness must be 0-100")
        payload = color_cmd(*rgb, brightness=args.brightness)
    elif args.action == "raw":
        try:
            payload = bytes.fromhex("".join(args.values))
        except ValueError:
            payload = b""
        if not payload:
            p.error("raw needs hex bytes, e.g. `raw CC 24 33`")
    asyncio.run(run(args.address, args.char, args.action, payload))


if __name__ == "__main__":
    main()
