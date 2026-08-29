"""Pytest configuration for gss-frontend tests."""

import sys

# Block torchcodec import to avoid FFmpeg library loading errors in CI
# torchcodec is only needed for advanced audio codecs, not for WAV files used in tests
sys.modules["torchcodec"] = None
