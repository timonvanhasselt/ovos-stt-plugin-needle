# OVOS STT Plugin Needle (Whistle)

An offline, privacy-respecting Speech-to-Text plugin for [Open Voice OS (OVOS)](https://openvoiceos.org/) powered by [Needle](https://github.com/cactus-compute/needle) and the **Whistle** model.

## Supported Languages

Supports 7 languages natively:
* English (`en`)
* German (`de`)
* French (`fr`)
* Spanish (`es`)
* Italian (`it`)
* Dutch (`nl`)
* Polish (`pl`)

## Installation

```bash
pip install ovos-stt-plugin-needle
```

## Configuration

Add the following configuration to your OVOS STT settings:

```json
{
  "stt": {
    "module": "ovos-stt-plugin-needle",
    "ovos-stt-plugin-needle": {
      "model": "cactus-compute/whistle",
      "device": "cpu"
    }
  }
}
```

## Sources & References

* [Cactus Compute - Needle & Whistle Repository](https://github.com/cactus-compute/needle#whistle)
* [Hugging Face - Needle 3 Model Repository](https://huggingface.co/Cactus-Compute/needle3)