# `map-gpt-embeddings` Singer Map Transformer

Inline mapper for splitting documents and calculating OpenAI embeddings, for purposes of building vectorstore knowledge base usable by GPT and ChatGPT.

Built with the [Meltano Tap SDK](https://sdk.meltano.com) for Singer Taps.

## See also

1. https://github.com/meltanolabs/tap-beautifulsoup - [Coming soon!] A tap for scraping web content from ReadTheDocs and other websites.
1. https://github.com/meltanolabs/target-chromadb - A Singer target that can be used to load documents and embeddings created by this library.
1. https://github.com/meltanolabs/gpt-ext - A Meltano utility which can be used to chat with the documents after they are loaded into the vector store.

## Prereqs

Before using this tap for ReadTheDocs, you first need to download the site locally using wget:

```console
% MY_RTD_SITE=sdk.meltano.com
% wget -r -A.html https://${MY_RTD_SITE}/en/latest/
```

<!--

Developer TODO: Update the below as needed to correctly describe the install procedure. For instance, if you do not have a PyPi repo, or if you want users to directly install from your git repo, you can modify this step as appropriate.

## Installation

Install from PyPi:

```bash
pipx install map-gpt-embeddings
```

Install from GitHub:

```bash
pipx install git+https://github.com/ORG_NAME/map-gpt-embeddings.git@main
```

-->

## Configuration

### Accepted Config Options

<!--
Developer TODO: Provide a list of config options accepted by the tap.

This section can be created by copy-pasting the CLI output from:

```
map-gpt-embeddings --about --format=markdown
```
-->

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

<!--
Developer TODO:
Your project comes with a custom `meltano.yml` project file already created. Open the `meltano.yml` and follow any "TODO" items listed in
the file.
-->

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
meltano elt map-gpt-embeddings target-jsonl
```

### SDK Dev Guide

See the [dev guide](https://sdk.meltano.com/en/latest/dev_guide.html) for more instructions on how to use the SDK to
develop your own taps and targets.
