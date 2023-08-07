# `map-gpt-embeddings`

Inline mapper for splitting documents and calculating OpenAI embeddings, for purposes of building vectorstore knowledge base usable by GPT and ChatGPT.
Split documents into segments, then vectorize.

Built with the [Meltano Singer SDK](https://sdk.meltano.com).

## Capabilities

* `stream-maps`

## Settings

| Setting                   | Required | Default | Description |
|:--------------------------|:--------:|:-------:|:------------|
| document_text_property    | False    | page_content |             |
| document_metadata_property| False    | metadata |             |
| openai_api_key            | False    | None    | OpenAI API key. Optional if `OPENAI_API_KEY` env var is set. |
| stream_maps               | False    | None    | Config object for stream maps capability. For more information check out [Stream Maps](https://sdk.meltano.com/en/latest/stream_maps.html). |
| stream_map_config         | False    | None    | User-defined config values to be used within map expressions. |

A full list of supported settings and capabilities is available by running: `map-openai-embeddings --about`

## See also

The demo project that originally used this mapper https://github.com/MeltanoLabs/gpt-meltano-demo.

## Configuration

### Accepted Config Options


A full list of supported settings and capabilities for this
tap is available by running:

```bash
map-gpt-embeddings --about
```

### Configure using environment variables

This Singer tap will automatically import any environment variables within the working directory's
`.env` if the `--config=ENV` is provided, such that config values will be considered if a matching
environment variable is set either in the terminal context or in the `.env` file.

### OpenAI Authentication and Authorization

You will need an OpenAI API Key to calculate embeddings using OpenAI's models. Free accounts are rate limited to 60 calls per minute. This is different from ChatGPT Plus account and requires a per-API call billing method established with OpenAI.

## Usage

You can easily run `map-gpt-embeddings` by itself or in a pipeline using [Meltano](https://meltano.com/).

### Executing the Tap Directly

```bash
map-gpt-embeddings --version
map-gpt-embeddings --help
map-gpt-embeddings --config CONFIG --discover > ./catalog.json
```

## Developer Resources

Follow these instructions to contribute to this project.

### Initialize your Development Environment

```bash
pipx install poetry
poetry install
```

### Create and Run Tests

Create tests within the `map_gpt_embeddings/tests` subfolder and
then run:

```bash
poetry run pytest
```

You can also test the `map-gpt-embeddings` CLI interface directly using `poetry run`:

```bash
poetry run map-gpt-embeddings --help
```

### Testing with [Meltano](https://www.meltano.com)

_**Note:** This tap will work in any Singer environment and does not require Meltano.
Examples here are for convenience and to streamline end-to-end orchestration scenarios._


Next, install Meltano (if you haven't already) and any needed plugins:

```bash
# Install meltano
pipx install meltano
# Initialize meltano within this directory
cd map-gpt-embeddings
meltano install
```

Now you can test and orchestrate using Meltano:

```bash
# Test invocation:
meltano invoke map-gpt-embeddings --version
# OR run a test `elt` pipeline:
meltano run tap-smoke-test map-gpt-embeddings target-jsonl
```

### SDK Dev Guide

See the [dev guide](https://sdk.meltano.com/en/latest/dev_guide.html) for more instructions on how to use the SDK to
develop your own taps and targets.
