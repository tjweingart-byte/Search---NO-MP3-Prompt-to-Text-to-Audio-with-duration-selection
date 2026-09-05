"""The FAM Experiment Engineer: an experiment layer around FAM, not inside it.

Nothing in this package is imported by production FAM. It imports *from*
production - `tts.py`, `script_generator.py`, `config.py` - so that what gets
measured is the code that ships rather than a copy of it, and the dependency
only ever points that way.
"""
