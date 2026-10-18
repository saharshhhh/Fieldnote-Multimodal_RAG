import asyncio
import json
import pathlib

import edge_tts

OUT = pathlib.Path("data/calls")
OUT.mkdir(parents=True, exist_ok=True)

VOICES = {"TECH": "en-US-GuyNeural", "SUPPORT": "en-US-JennyNeural"}

# All numbers below are FICTIONAL, written for a synthetic eval corpus.
CALLS = {
    "call_01_ac_frozen_coil": {
        "lines": [
            ("SUPPORT", "Technical support, this is Dana. How can I help?"),
            ("TECH", "Hi Dana. I'm at a house with a central air conditioner where the indoor coil keeps freezing over."),
            ("SUPPORT", "Check airflow first. When was the filter last changed, and is the blower running?"),
            ("TECH", "The filter is clogged solid. The blower is running fine."),
            ("SUPPORT", "Replace the filter, then shut the compressor off and let the coil thaw completely. That usually takes about four hours."),
            ("TECH", "And once it's running again, what should I verify?"),
            ("SUPPORT", "The temperature drop across the coil should be between sixteen and twenty degrees. Under sixteen with good airflow points to low charge."),
            ("TECH", "Got it. Thaw four hours, new filter, then check the sixteen to twenty degree drop."),
        ],
        "facts": [
            ("How long should a frozen indoor coil thaw with the compressor off?", "About four hours."),
            ("What temperature drop across the coil is expected on an air conditioner?", "Between sixteen and twenty degrees."),
        ],
    },
    "call_02_ac_capacitor": {
        "lines": [
            ("SUPPORT", "Technical support, Marcus speaking."),
            ("TECH", "Marcus, the outdoor fan isn't turning and the compressor just hums. I suspect the dual run capacitor."),
            ("SUPPORT", "Likely. Read the rating on the label. If the measured value is more than six percent below rated, replace it."),
            ("TECH", "The label says forty plus five microfarads. My meter reads thirty-three on the compressor side."),
            ("SUPPORT", "Thirty-three against forty is well out of tolerance. Replace it with the same rating and the same voltage, four hundred forty volts."),
            ("TECH", "Same rating, four forty volts. Understood."),
            ("SUPPORT", "If you spin the fan blade by hand and it starts running, that confirms the capacitor is the problem."),
        ],
        "facts": [
            ("How far below its rating can a run capacitor measure before replacement?", "More than six percent below rated means replace it."),
            ("What voltage rating should the replacement dual run capacitor have?", "Four hundred forty volts, same rating as the original."),
        ],
    },
    "call_03_heat_pump_defrost": {
        "lines": [
            ("SUPPORT", "Technical support, this is Dana."),
            ("TECH", "Dana, the outdoor unit of a heat pump is covered in ice this morning. Is that normal?"),
            ("SUPPORT", "Frost is normal. A solid block of ice is not. The control board should start a defrost cycle about every ninety minutes of runtime when the coil sensor is below thirty-two degrees."),
            ("TECH", "How long should a defrost cycle last?"),
            ("SUPPORT", "Up to ten minutes. Steam rising off the unit during defrost is normal."),
            ("TECH", "And if it never goes into defrost?"),
            ("SUPPORT", "Check the defrost sensor first, then the control board."),
        ],
        "facts": [
            ("How often should a heat pump enter defrost?", "About every ninety minutes of runtime."),
            ("How long should a defrost cycle last?", "Up to ten minutes."),
        ],
    },
    "call_04_heat_pump_thermostat_ob": {
        "lines": [
            ("SUPPORT", "Technical support, Marcus speaking."),
            ("TECH", "I installed a new thermostat on a heat pump and it blows cold air in heating mode."),
            ("SUPPORT", "That is the reversing valve setting. Check the unit's wiring label. If the valve energizes in cooling, set the thermostat to O. If it energizes in heating, set it to B."),
            ("TECH", "The label says energized in cooling."),
            ("SUPPORT", "Then set O. While you are in setup, set the auxiliary heat lockout at forty degrees outdoors, so the electric strips don't run when the heat pump can handle the load."),
            ("TECH", "O for the valve, and aux lockout at forty degrees."),
        ],
        "facts": [
            ("Which reversing valve setting is used if the valve energizes in cooling?", "Set the thermostat to O."),
            ("What outdoor temperature should the auxiliary heat lockout be set to?", "Forty degrees."),
        ],
    },
    "call_05_furnace_flame_sensor": {
        "lines": [
            ("SUPPORT", "Technical support, this is Dana."),
            ("TECH", "Dana, a gas furnace lights the burners for a few seconds and then shuts down. The board is flashing a fault code."),
            ("SUPPORT", "Burners lighting then dropping out is almost always a dirty flame sensor. Turn off power and gas, remove the sensor, and clean it."),
            ("TECH", "Should I use sandpaper?"),
            ("SUPPORT", "No. Use a fine abrasive pad with a light touch. Then check the flame signal. You want at least one microamp, and below zero point five microamps the board will lock out."),
            ("TECH", "Fine pad, no sandpaper, and at least one microamp."),
        ],
        "facts": [
            ("What should you clean a furnace flame sensor with?", "A fine abrasive pad with a light touch, not sandpaper."),
            ("What flame signal is needed for a healthy flame sensor?", "At least one microamp; below zero point five microamps the board locks out."),
        ],
    },
    "call_06_furnace_pressure_switch": {
        "lines": [
            ("SUPPORT", "Technical support, Marcus speaking."),
            ("TECH", "Marcus, the inducer motor runs on a furnace but the burners never ignite."),
            ("SUPPORT", "The pressure switch is probably not closing. Check the vent for blockage, the tubing for water or cracks, and the condensate drain on a high efficiency unit."),
            ("TECH", "There is water sitting in the pressure switch tubing."),
            ("SUPPORT", "Clear it and check the condensate trap. Also confirm the inducer is spinning at full speed. And never jumper the pressure switch to get the furnace running."),
            ("TECH", "Clear the tubing, check the trap, and never jumper the switch."),
        ],
        "facts": [
            ("What should you never do when a furnace pressure switch will not close?", "Never jumper the pressure switch."),
            ("What three things should be checked when the pressure switch does not close?", "The vent for blockage, the tubing for water or cracks, and the condensate drain."),
        ],
    },
    "call_07_thermostat_no_c_wire": {
        "lines": [
            ("SUPPORT", "Technical support, this is Dana."),
            ("TECH", "Dana, a smart thermostat keeps rebooting and dropping off Wi-Fi."),
            ("SUPPORT", "Most likely there is no common wire. Look for a spare conductor in the cable and connect it as C at both the thermostat and the furnace control board."),
            ("TECH", "The cable has four wires and all four are already in use."),
            ("SUPPORT", "Then use the add-a-wire adapter kit, or run a new cable. Do not jumper R to C, because that can blow the transformer fuse."),
            ("TECH", "Adapter kit, no jumper from R to C."),
        ],
        "facts": [
            ("What should you do for a thermostat with no spare wire for C?", "Use an add-a-wire adapter kit or run a new cable."),
            ("Why should you not jumper R to C?", "It can blow the transformer fuse."),
        ],
    },
    "call_08_air_purifier_service": {
        "lines": [
            ("SUPPORT", "Technical support, Marcus speaking."),
            ("TECH", "A customer asks how often to service their whole home air purifier."),
            ("SUPPORT", "Replace the filter cartridge every twelve months, and the ultraviolet lamp every twenty-four months."),
            ("TECH", "Is there anything else that affects performance?"),
            ("SUPPORT", "Yes. Make sure the cabinet door is sealed. A cracked gasket lets unfiltered air bypass the purifier."),
            ("TECH", "Cartridge yearly, lamp every two years, and check the door seal."),
        ],
        "facts": [
            ("How often should the air purifier filter cartridge be replaced?", "Every twelve months."),
            ("How often should the ultraviolet lamp be replaced?", "Every twenty-four months."),
        ],
    },
    "call_09_iaq_humidifier_setpoint": {
        "lines": [
            ("SUPPORT", "Technical support, this is Dana."),
            ("TECH", "Dana, a customer has condensation on all the windows in winter and a whole home humidifier."),
            ("SUPPORT", "Set the humidity target by outdoor temperature. At twenty degrees outdoors, keep indoor humidity near thirty percent. Above freezing you can go up to forty percent."),
            ("TECH", "The customer has it set to fifty percent."),
            ("SUPPORT", "That is too high. Lower it to thirty percent and the windows should clear within a few days."),
            ("TECH", "Thirty percent at twenty degrees outdoors. Got it."),
        ],
        "facts": [
            ("What indoor humidity target is right when it is twenty degrees outdoors?", "About thirty percent."),
            ("What is the maximum indoor humidity target when the outdoor temperature is above freezing?", "Forty percent."),
        ],
    },
    "call_10_iaq_erv_ventilation": {
        "lines": [
            ("SUPPORT", "Technical support, Marcus speaking."),
            ("TECH", "Marcus, a tight new home feels stuffy and the carbon dioxide reading is fifteen hundred parts per million."),
            ("SUPPORT", "Target under one thousand parts per million. An energy recovery ventilator brings in fresh air and recovers heat from the exhaust."),
            ("TECH", "Where should the fresh air duct connect?"),
            ("SUPPORT", "To the return plenum, ahead of the air filter, so the incoming air is filtered."),
            ("TECH", "Under one thousand, and connect ahead of the filter."),
        ],
        "facts": [
            ("What carbon dioxide level should indoor air stay under?", "Under one thousand parts per million."),
            ("Where should an energy recovery ventilator's fresh air duct connect?", "To the return plenum, ahead of the air filter."),
        ],
    },
}


async def render_call(name: str, lines):
    tmp = OUT / f"_{name}"
    tmp.mkdir(exist_ok=True)
    parts = []
    for i, (speaker, text) in enumerate(lines):
        p = tmp / f"{i:03d}.mp3"
        await edge_tts.Communicate(text, VOICES[speaker], rate="-5%").save(str(p))
        parts.append(p)
    # MP3 frames can be joined byte-for-byte; Whisper decodes the result fine
    with open(OUT / f"{name}.mp3", "wb") as out:
        for p in parts:
            out.write(p.read_bytes())
    for p in parts:
        p.unlink()
    tmp.rmdir()
    print("wrote", OUT / f"{name}.mp3")


def write_gold_facts():
    gold = []
    for name, spec in CALLS.items():
        for question, answer in spec["facts"]:
            gold.append({
                "source": f"{name}.mp3",
                "modality": "audio",
                "question": question,
                "gold_answer": answer,
            })
    path = OUT / "gold_facts.json"
    path.write_text(json.dumps(gold, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote", path, f"({len(gold)} audio eval facts)")


async def main():
    for name, spec in CALLS.items():
        await render_call(name, spec["lines"])
    write_gold_facts()


if __name__ == "__main__":
    asyncio.run(main())