"""drlib — shared internals for the music-restoration tools.

Not a public API; the CLI entry points are the scripts next to this package.
Everything environment-specific (phone device, music folder, work dirs) lives
in drlib.config: env var > ~/.config/music-restoration/config.env > auto-detect.
"""
