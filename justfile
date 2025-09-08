@_default:
    just --list

# Build container image
@image:
    podman build -t tree-evolution:latest .

# Format the code using ruff
@format:
    uv run ruff format

# Lint the code using ruff
@lint:
    uv run ruff check

# Order imports using ruff
@fix-imports:
    uv run ruff check --select I001 --fix

# Remove all files not tracked by git
@clean:
    git clean -f -x -d

# Start a shell on a running container
@shell:
    podman exec \
        --interactive \
        --tty \
        --env "TERM=xterm-256color" \
    tree-evolution-jupyter bash

# Start jupyter server in a container
@jupyter:
    podman run \
        --rm \
        --interactive \
        --tty \
        --detach \
        -p 8888:8888 \
        --name tree-evolution-jupyter \
        --volume .:/code:z \
        tree-evolution:latest \
        uv run jupyter lab \
            --ip 0.0.0.0 \
            --allow-root \
            --no-browser

# Kill jupyter server container
@jupyter-kill:
    podman kill tree-evolution-jupyter

# Open jupyter lab in a browser
browser:
    #!/usr/bin/env bash
    set -euo pipefail

    token=$(
        podman exec tree-evolution-jupyter      \
            uv run jupyter lab list --json \
        | jq -r ".token"
    )
    url="http://127.0.0.1:8888/lab?token=${token}"
    xdg-open "${url}"


# Generate notebooks from sources
@gen-notebooks:
    uv run jupytext --sync $(find jupytext -type f)

# Install git hooks
@install-hooks:
    cp hooks/* .git/hooks/

# Prepare all the necessary files
@setup:
    podman run \
        --rm \
        --interactive \
        --tty \
        --volume .:/code:z \
        hp-rvpinn:latest \
        just \
            install-hooks \
            gen-notebooks
