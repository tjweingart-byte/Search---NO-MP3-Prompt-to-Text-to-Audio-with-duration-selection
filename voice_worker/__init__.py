"""The GPU half of FAM's voice, deployable on its own.

`synth.py` does the work; `handler.py` and `server.py` are two envelopes around
it - RunPod Serverless and a plain HTTP port. Nothing in here is imported by
the app: `remote_voice.py` reaches it over the network, which is the point.
"""
