<img width="128" height="128" align="right" alt="Logo" src="https://github.com/user-attachments/assets/df5f2840-2f92-4991-aa57-252747d7182e" />

# Heretic: Fully automatic censorship removal for language models<br><br>[![Discord](https://img.shields.io/discord/1447831134212984903?color=5865F2&label=discord&labelColor=black&logo=discord&logoColor=white&style=for-the-badge)](https://discord.gg/gdXc48gSyT) [![Follow us on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/follow-us-on-hf-md-dark.svg)](https://huggingface.co/heretic-org)

[![#1 Repository of the Day](https://trendshift.io/api/badge/repositories/20538)](https://trendshift.io/repositories/20538)

Heretic is a tool that removes censorship (aka "safety alignment") from
transformer-based language models without expensive post-training.
It combines an advanced implementation of directional ablation, also known
as "abliteration" ([Arditi et al. 2024](https://arxiv.org/abs/2406.11717),
Lai 2025 ([1](https://huggingface.co/blog/grimjim/projected-abliteration),
[2](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration))),
with a TPE-based parameter optimizer powered by [Optuna](https://optuna.org/).

This approach enables Heretic to work **completely automatically.** Heretic
finds high-quality abliteration parameters by co-minimizing the number of
refusals and the KL divergence from the original model. This results in a
decensored model that retains as much of the original model's intelligence
as possible. Using Heretic does not require an understanding of transformer
internals. In fact, anyone who knows how to run a command-line program
can use Heretic to decensor language models.

<img width="650" height="715" alt="Screenshot" src="https://github.com/user-attachments/assets/d71a5efa-d6be-4705-a817-63332afb2d15" />

&nbsp;

Running unsupervised with the default configuration, Heretic can produce
decensored models that rival the quality of abliterations created manually
by human experts:

| Model | Refusals for "harmful" prompts | KL divergence from original model for "harmless" prompts |
| :--- | ---: | ---: |
| [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) (original) | 97/100 | 0 *(by definition)* |
| [mlabonne/gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2) | 3/100 | 1.04 |
| [huihui-ai/gemma-3-12b-it-abliterated](https://huggingface.co/huihui-ai/gemma-3-12b-it-abliterated) | 3/100 | 0.45 |
| **[p-e-w/gemma-3-12b-it-heretic](https://huggingface.co/p-e-w/gemma-3-12b-it-heretic) (ours)** | **3/100** | **0.16** |

The Heretic version, generated without any human effort, achieves the same
level of refusal suppression as other abliterations, but at a much lower
KL divergence, indicating less damage to the original model's capabilities.
*(You can reproduce those numbers using Heretic's built-in evaluation functionality,
e.g. `heretic --model google/gemma-3-12b-it --evaluate-model p-e-w/gemma-3-12b-it-heretic`.
Note that the exact values might be platform- and hardware-dependent.
The table above was compiled using PyTorch 2.8 on an RTX 5090.)*

Of course, mathematical metrics and automated benchmarks never tell the whole
story, and are no substitute for human evaluation. Models generated with
Heretic have been well-received by users (links and emphasis added):

> "I was skeptical before, but I just downloaded
> [**GPT-OSS 20B Heretic**](https://huggingface.co/p-e-w/gpt-oss-20b-heretic)
> model and holy shit. It gives properly formatted long responses to sensitive topics,
> using the exact uncensored words that you would expect from an uncensored model,
> produces markdown format tables with details and whatnot. Looks like this is
> the best abliterated version of this model so far..."
> [*(Link to comment)*](https://old.reddit.com/r/LocalLLaMA/comments/1oymku1/heretic_fully_automatic_censorship_removal_for/np6tba6/)

> "[**Heretic GPT 20b**](https://huggingface.co/p-e-w/gpt-oss-20b-heretic)
> seems to be the best uncensored model I have tried yet. It doesn't destroy a
> the model's intelligence and it is answering prompts normally would be
> rejected by the base model."
> [*(Link to comment)*](https://old.reddit.com/r/LocalLLaMA/comments/1oymku1/heretic_fully_automatic_censorship_removal_for/npe9jng/)

> "[[**Qwen3-4B-Instruct-2507-heretic**](https://huggingface.co/p-e-w/Qwen3-4B-Instruct-2507-heretic)]
> Has been the best unquantized abliterated model that I have been able to run on 16gb vram."
> [*(Link to comment)*](https://old.reddit.com/r/LocalLLaMA/comments/1phjxca/im_calling_these_people_out_right_now/nt06tji/)

Heretic supports most dense models, including many multimodal models, and
several different MoE architectures. It does not yet support SSMs/hybrid models,
models with inhomogeneous layers, and certain novel attention systems.

You can find a collection of models that have been decensored using Heretic
[on Hugging Face](https://huggingface.co/collections/p-e-w/the-bestiary),
and the community has created and published
[well over 1,000](https://huggingface.co/models?other=heretic)
Heretic models in addition to those.

## Documentation

- **[Installation & Usage](#installation)** - Getting started guide (below)
- **[DOCKER.md](DOCKER.md)** - Comprehensive Docker deployment guide
- **[DEVELOPMENT.md](DEVELOPMENT.md)** - Contributing and development guide

## Installation

Heretic requires Python 3.10+ and PyTorch 2.2+ with appropriate GPU support for your hardware.

**Note:** This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python dependency management. We recommend using one of the installation methods below rather than installing with pip.

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/p-e-w/heretic.git
cd heretic

# Install dependencies (includes optional research features)
uv sync --all-extras

# Run Heretic
uv run heretic Qwen/Qwen3-4B-Instruct-2507
```

### Option 2: Using Conda

```bash
# Clone the repository
git clone https://github.com/p-e-w/heretic.git
cd heretic

# Create and activate the Conda environment
conda env create -f environment.yml
conda activate heretic

# Install Heretic in development mode
pip install -e .

# Run Heretic
heretic Qwen/Qwen3-4B-Instruct-2507
```

**Note:** Adjust the CUDA version in `environment.yml` to match your system (e.g., cuda=11.8 or cuda=12.1).

### Option 3: Using Docker

**Prerequisites:** Docker with NVIDIA GPU support ([nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

```bash
# Clone the repository
git clone https://github.com/p-e-w/heretic.git
cd heretic

# Build and run with Docker Compose (recommended)
docker-compose run heretic heretic Qwen/Qwen3-4B-Instruct-2507

# Or build and run with Docker directly
docker build -t heretic .
docker run --gpus all -it heretic heretic Qwen/Qwen3-4B-Instruct-2507
```

For detailed Docker usage, configuration, and troubleshooting, see **[DOCKER.md](DOCKER.md)**.

Replace `Qwen/Qwen3-4B-Instruct-2507` with whatever model you want to decensor.

## Usage

Once installed, simply run:

```bash
heretic MODEL_NAME
```

The process is fully automatic and does not require configuration; however,
Heretic has a variety of configuration parameters that can be changed for
greater control. Run `heretic --help` to see available command-line options,
or look at [`config.default.toml`](config.default.toml) if you prefer to use
a configuration file.

At the start of a program run, Heretic benchmarks the system to determine
the optimal batch size to make the most of the available hardware.
On an RTX 3090, with the default configuration, decensoring Llama-3.1-8B-Instruct
takes about 45 minutes. Note that Heretic supports model quantization with
bitsandbytes, which can drastically reduce the amount of VRAM required to process
models. Set the `quantization` option to `bnb_4bit` to enable quantization.

After Heretic has finished decensoring a model, you are given the option to
save the model, upload it to Hugging Face, chat with it to test how well it works,
or any combination of those actions.


## Research features

In addition to its primary function of removing model censorship, Heretic also
provides features designed to support research into the semantics of model internals
(interpretability). To use those features, install Heretic with the optional
`research` extra:

```bash
pip install -U heretic-llm[research]
# Or, if using uv:
uv sync --all-extras
```

This gives you access to the following functionality:

### Generate plots of residual vectors by passing `--plot-residuals`

When run with this flag, Heretic will:

1. Compute residual vectors (hidden states) for the first output token,
   for each transformer layer, for both "harmful" and "harmless" prompts.
2. Perform a [PaCMAP projection](https://github.com/YingfanWang/PaCMAP)
   from residual space to 2D-space.
3. Left-right align the projections of "harmful"/"harmless" residuals
   by their geometric medians to make projections for consecutive layers
   more similar. Additionally, PaCMAP is initialized with the previous
   layer's projections for each new layer, minimizing disruptive transitions.
4. Scatter-plot the projections, generating a PNG image for each layer.
5. Generate an animation showing how residuals transform between layers,
   as an animated GIF.

<img width="800" height="600" alt="Plot of residual vectors" src="https://github.com/user-attachments/assets/981aa6ed-5ab9-48f0-9abf-2b1a2c430295" />

See [the configuration file](config.default.toml) for options that allow you
to control various aspects of the generated plots.

Note that PaCMAP is an expensive operation that is performed on the CPU.
For larger models, it can take an hour or more to compute projections
for all layers.

### Print details about residual geometry by passing `--print-residual-geometry`

If you are interested in a quantitative analysis of how residual vectors
for "harmful" and "harmless" prompts relate to each other, this flag gives you
a rich table with per-layer metrics including cosine similarities between residual
directions, L2 norms, and the mean silhouette coefficient for the good/bad clusters.


## How Heretic works

Heretic implements a parametrized variant of directional ablation. For each
supported transformer component (currently, attention out-projection and
MLP down-projection), it identifies the associated matrices in each transformer
layer, and orthogonalizes them with respect to the relevant "refusal direction",
inhibiting the expression of that direction in the result of multiplications
with that matrix.

Refusal directions are computed for each layer as a difference-of-means between
the first-token residuals for "harmful" and "harmless" example prompts.

The ablation process is controlled by several optimizable parameters:

* `direction_index`: Either the index of a refusal direction, or the special
  value `per layer`, indicating that each layer should be ablated using the
  refusal direction associated with that layer.
* `max_weight`, `max_weight_position`, `min_weight`, and `min_weight_distance`:
  For each component, these parameters describe the shape and position of the
  ablation weight kernel over the layers. The following diagram illustrates this:

<img width="800" height="500" alt="Explanation" src="https://github.com/user-attachments/assets/82e4b84e-5a82-4faf-b918-ac642f9e4892" />

&nbsp;

Heretic's main innovations over existing abliteration systems are:

* The shape of the ablation weight kernel is highly flexible, which, combined with
  automatic parameter optimization, can improve the compliance/quality tradeoff.
  Non-constant ablation weights were previously explored by Maxime Labonne in
  [gemma-3-12b-it-abliterated-v2](https://huggingface.co/mlabonne/gemma-3-12b-it-abliterated-v2).
* The refusal direction index is a float rather than an integer. For non-integral
  values, the two nearest refusal direction vectors are linearly interpolated.
  This unlocks a vast space of additional directions beyond the ones identified
  by the difference-of-means computation, and often enables the optimization
  process to find a better direction than that belonging to any individual layer.
* Ablation parameters are chosen separately for each component. I have found that
  MLP interventions tend to be more damaging to the model than attention interventions,
  so using different ablation weights can squeeze out some extra performance.


## Prior art

I'm aware of the following publicly available implementations of abliteration
techniques:

* [AutoAbliteration](https://huggingface.co/posts/mlabonne/714992455492422)
* [abliterator.py](https://github.com/FailSpy/abliterator)
* [wassname's Abliterator](https://github.com/wassname/abliterator)
* [ErisForge](https://github.com/Tsadoq/ErisForge)
* [Removing refusals with HF Transformers](https://github.com/Sumandora/remove-refusals-with-transformers)
* [deccp](https://github.com/AUGMXNT/deccp)

Note that Heretic was written from scratch, and does not reuse code from
any of those projects.


## Acknowledgments

The development of Heretic was informed by:

* [The original abliteration paper (Arditi et al. 2024)](https://arxiv.org/abs/2406.11717)
* [Maxime Labonne's article on abliteration](https://huggingface.co/blog/mlabonne/abliteration),
  as well as some details from the model cards of his own abliterated models (see above)
* [Jim Lai's article describing "projected abliteration"](https://huggingface.co/blog/grimjim/projected-abliteration)
  and ["norm-preserving biprojected abliteration"](https://huggingface.co/blog/grimjim/norm-preserving-biprojected-abliteration)


## Citation

If you use Heretic for your research, please cite it using the following BibTeX entry:

```bibtex
@misc{heretic,
  author = {Weidmann, Philipp Emanuel},
  title = {Heretic: Fully automatic censorship removal for language models},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/p-e-w/heretic}}
}
```


## License

Copyright &copy; 2025-2026  Philipp Emanuel Weidmann (<pew@worldwidemann.com>) + contributors

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

**By contributing to this project, you agree to release your
contributions under the same license.**
