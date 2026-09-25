# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Real unit tests for ImageAttachment / AgentTaskRequest.images validation
(src/api/models/requests.py) — pure Pydantic, no I/O.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.models.requests import AgentTaskRequest, ImageAttachment


def test_image_attachment_accepts_a_real_supported_media_type():
    img = ImageAttachment(media_type="image/png", data="aGVsbG8=")
    assert img.media_type == "image/png"
    assert img.data == "aGVsbG8="


def test_image_attachment_rejects_an_unsupported_media_type():
    with pytest.raises(ValidationError, match="media_type must be one of"):
        ImageAttachment(media_type="application/pdf", data="aGVsbG8=")


def test_image_attachment_rejects_empty_data():
    with pytest.raises(ValidationError):
        ImageAttachment(media_type="image/png", data="")


def test_image_attachment_rejects_data_over_the_size_cap():
    oversized = "a" * 15_000_001
    with pytest.raises(ValidationError):
        ImageAttachment(media_type="image/png", data=oversized)


def test_agent_task_request_accepts_no_images():
    req = AgentTaskRequest(task="A task description long enough to pass validation.")
    assert req.images is None


def test_agent_task_request_accepts_up_to_four_images():
    images = [ImageAttachment(media_type="image/png", data="aGVsbG8=") for _ in range(4)]
    req = AgentTaskRequest(task="A task description long enough to pass validation.", images=images)
    assert len(req.images) == 4


def test_agent_task_request_rejects_more_than_four_images():
    images = [ImageAttachment(media_type="image/png", data="aGVsbG8=") for _ in range(5)]
    with pytest.raises(ValidationError):
        AgentTaskRequest(task="A task description long enough to pass validation.", images=images)


def test_agent_task_request_rejects_an_unsupported_image_type_inside_the_list():
    with pytest.raises(ValidationError):
        AgentTaskRequest(
            task="A task description long enough to pass validation.",
            images=[{"media_type": "video/mp4", "data": "aGVsbG8="}],
        )
