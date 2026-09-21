"""
codec — frozen CelebA-HQ VQ-f4 and historical rejected codec backends.

Public surface:

    BaseCodec            Abstract codec contract (codec.base)
    CodecCheckpointError Raised on any metadata/schema failure
    load_codec           Backend-neutral factory (codec.codec_factory)

All other symbols are internal to their respective modules.
"""
from codec.base import BaseCodec, CodecCheckpointError, validate_checkpoint_metadata
from codec.codec_factory import load_codec

__all__ = [
    "BaseCodec",
    "CodecCheckpointError",
    "validate_checkpoint_metadata",
    "load_codec",
]
