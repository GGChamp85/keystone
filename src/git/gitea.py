# Copyright 2024-2026 Gaurav Gupta / Keystone
# Licensed under the Apache License, Version 2.0

"""
Keystone Agents — Gitea GitHost implementation.

Field names and auth scheme verified against Gitea's own published swagger
spec (github.com/go-gitea/gitea, templates/swagger/v1_json.tmpl) for the
v1.22 release line, not assumed: `CreatePullRequestOption` has
head/base/title/body; the `PullRequest` response's index/url/html_url;
token auth is `Authorization: token <token>` (the `AuthorizationHeaderToken`
security scheme — the older `?token=` query-param and `token <token>`-less
`Token` schemes are deprecated for removal and deliberately not used here).
"""

from __future__ import annotations

import httpx
import structlog

from src.config import get_settings
from src.git.host import GitHostError, PullRequestInfo, PullRequestStatus

logger = structlog.get_logger(__name__)


class GiteaHost:
    name = "gitea"

    def __init__(self) -> None:
        settings = get_settings()
        token = settings.git_host_token.get_secret_value() if settings.git_host_token else None
        headers = {"Authorization": f"token {token}"} if token else {}
        self._client = httpx.AsyncClient(
            base_url=settings.git_host_api_url.rstrip("/"),
            headers=headers,
            timeout=30.0,
        )

    async def get_default_branch(self, owner: str, repo: str) -> str:
        resp = await self._client.get(f"/repos/{owner}/{repo}")
        self._raise_for_status(resp, f"GET /repos/{owner}/{repo}")
        return resp.json()["default_branch"]

    async def open_pull_request(
        self,
        owner: str,
        repo: str,
        *,
        head: str,
        base: str,
        title: str,
        body: str,
    ) -> PullRequestInfo:
        resp = await self._client.post(
            f"/repos/{owner}/{repo}/pulls",
            json={"head": head, "base": base, "title": title, "body": body},
        )
        self._raise_for_status(resp, f"POST /repos/{owner}/{repo}/pulls")
        data = resp.json()
        logger.info("gitea.pr_opened", owner=owner, repo=repo, number=data["number"], url=data["html_url"])
        return PullRequestInfo(number=data["number"], url=data["url"], html_url=data["html_url"])

    async def get_pull_request_status(self, owner: str, repo: str, number: int) -> PullRequestStatus:
        resp = await self._client.get(f"/repos/{owner}/{repo}/pulls/{number}")
        self._raise_for_status(resp, f"GET /repos/{owner}/{repo}/pulls/{number}")
        data = resp.json()
        state = data["state"]
        if data.get("merged"):
            state = "merged"
        return PullRequestStatus(
            number=number,
            state=state,
            merged=bool(data.get("merged")),
            merge_commit_sha=data.get("merge_commit_sha"),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _raise_for_status(self, resp: httpx.Response, context: str) -> None:
        if resp.status_code >= 400:
            raise GitHostError(f"Gitea API error on {context}: {resp.status_code} {resp.text}")
