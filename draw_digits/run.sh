#!/usr/bin/env bash
blueprint-compiler compile --output main_window.ui ./main_window.blp
uv run main.py
