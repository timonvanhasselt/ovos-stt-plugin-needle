from setuptools import setup, find_packages

setup(
    name="ovos-stt-plugin-needle",
    version="1.0.1",
    description="Offline STT plugin for OVOS using Cactus Needle Whistle model",
    author="Your Name",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=[
        "ovos-plugin-manager>=0.0.1",
        "cactus-needle",
        "huggingface-hub",
        "numpy",
    ],
    entry_points={
        "opm.stt": [
            "ovos-stt-plugin-needle = ovos_stt_plugin_needle:NeedleSTTPlugin"
        ]
    },
)