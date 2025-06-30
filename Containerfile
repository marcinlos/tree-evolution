FROM ubuntu:24.04

WORKDIR /code

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /uvx /bin/

ENV DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        software-properties-common \
        git \
        python3 \
        python3-pip \
        just

# Set up the project dependencies
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/venv

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,z \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,z \
    uv venv && \
    uv sync --frozen --no-install-project

COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN uv run pre-commit install-hooks
