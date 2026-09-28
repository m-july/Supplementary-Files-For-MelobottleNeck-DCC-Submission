# simplemono_preproc/__init__.py
"""
SimpleMono preprocessing toolkit.

Separation of concerns:
1) score_readers: MIDI / MusicXML / Kern -> unified notes (pitch, start_pos, end_pos)
2) encoder: notes -> SimpleMono triples / token ids
3) splitting + pipelines: build datasets for different tasks without leaking labels
"""

from .vocab import SimpleMonoVocab
