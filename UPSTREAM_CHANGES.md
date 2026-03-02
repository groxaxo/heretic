# Upstream Changes Report

**Comparison:** `groxaxo/heretic:master` → `p-e-w/heretic:master`  
**Reference URL:** https://github.com/groxaxo/heretic/compare/master...p-e-w%3Aheretic%3Amaster  
**Commits ahead:** 63 commits  
**Files changed:** 16  
**Version bump:** `1.0.1` → `1.2.0`

---

## Summary

The upstream `p-e-w/heretic` repository has undergone major improvements across all files. These changes are **highly recommended** to implement because they improve performance, correctness, compatibility, and user experience. The core architecture was changed from direct in-place weight modification to a **LoRA (PEFT) adapter-based** approach, which is both faster and more flexible.

---

## Decision: ✅ Implement All Changes

All upstream changes are worth implementing. Reasons:

1. **LoRA-based abliteration** significantly reduces memory usage and speeds up trial reloading.
2. **4-bit quantization** (bitsandbytes) allows processing of much larger models on limited VRAM.
3. **Checkpoint/resume** prevents loss of progress in long runs.
4. **Multimodal model support** broadens compatibility.
5. **Research features** (analyzer) enable deeper insight into the abliteration process.
6. **Row normalization** and **orthogonalized directions** improve abliteration quality.
7. **Winsorization** handles models with "massive activations".
8. **Better UX** (notebook support, memory display, interactive prompts).

The fork added a **vLLM integration** that the upstream chose not to include; this is removed as part of the sync.

---

## Detailed Changes by File

### 1. `pyproject.toml` — Dependency and Version Updates

| Change | Description |
|--------|-------------|
| Version | `1.0.1` → `1.2.0` |
| Added | `bitsandbytes~=0.45` — 4-bit quantization support |
| Added | `peft~=0.14` — LoRA adapter framework |
| Added | `psutil~=7.1` — memory usage monitoring |
| Added | `kernels~=0.11` — kernels support |
| Changed | `accelerate>=1.10.0` → `accelerate~=1.10` |
| Changed | `datasets>=4.0.0` → `datasets~=4.0` |
| Changed | `transformers>=4.55.2` → `transformers~=4.57` |
| Changed | Pinned all deps with `~=` (compatible release) instead of `>=` |
| Removed | `vllm` optional dependency |
| Added | `[research]` optional extras: `geom-median`, `imageio`, `matplotlib`, `numpy`, `pacmap`, `scikit-learn` |
| Changed | `Changelog` URL updated |

### 2. `config.default.toml` — Configuration File Overhaul

The configuration file was completely rewritten. New options added:

| New Option | Default | Description |
|------------|---------|-------------|
| `quantization` | `"none"` | Quantization method (`"none"` or `"bnb_4bit"`) |
| `max_memory` | (commented out) | Per-device memory limits |
| `trust_remote_code` | (not in default) | Whether to trust remote code |
| `print_responses` | `false` | Print prompt/response pairs during evaluation |
| `print_residual_geometry` | `false` | Print detailed residual geometry analysis |
| `plot_residuals` | `false` | Generate PaCMAP residual vector plots |
| `residual_plot_path` | `"plots"` | Base path for residual plots |
| `residual_plot_title` | (long string) | Title for residual plots |
| `residual_plot_style` | `"dark_background"` | Matplotlib style for plots |
| `kl_divergence_target` | `0.01` | Target KL divergence (below this, refusal count is optimized) |
| `orthogonalize_direction` | `false` | Use orthogonalized refusal directions |
| `row_normalization` | `"none"` | Row normalization strategy (`"none"`, `"pre"`, `"full"`) |
| `full_normalization_lora_rank` | `3` | LoRA rank for full normalization mode |
| `winsorization_quantile` | `1.0` | Winsorization quantile for massive activations (1.0 = disabled) |
| `study_checkpoint_dir` | `"checkpoints"` | Directory for study checkpoints |
| `bfloat16` | (new dtype) | Added to fallback dtype list |

Removed options (from the fork):
- `inference_backend`
- `vllm_gpu_memory_utilization`
- `vllm_max_model_len`

Dataset specifications enhanced with:
- `residual_plot_label` — label for plot legend
- `residual_plot_color` — matplotlib color for plot

Refusal markers expanded to include contractions without apostrophes (e.g., `"i cant"`, `"i wont"`, `"im unable"`) for broader coverage.

### 3. `src/heretic/config.py` — Settings Class Overhaul

**Before (fork):** Simple settings with basic fields; used `SettingsConfigDict`.

**After (upstream):** Rich settings with full validation; uses proper `settings_customise_sources` with `CliSettingsSource` and `EnvSettingsSource`.

New types/enums:
```python
class QuantizationMethod(str, Enum):
    NONE = "none"
    BNB_4BIT = "bnb_4bit"

class RowNormalization(str, Enum):
    NONE = "none"
    PRE = "pre"
    FULL = "full"
```

`DatasetSpecification` now has:
- `prefix` — text to prepend to each prompt
- `suffix` — text to append to each prompt
- `system_prompt` — per-dataset system prompt override
- `residual_plot_label` — plot legend label
- `residual_plot_color` — plot color

New `Settings` fields:
- `quantization: QuantizationMethod` (replaces the fork's `str | None`)
- `trust_remote_code: bool | None`
- `max_memory: Dict[str, str] | None`
- `print_responses: bool`
- `print_residual_geometry: bool`
- `plot_residuals: bool`
- `residual_plot_path: str`
- `residual_plot_title: str`
- `residual_plot_style: str`
- `kl_divergence_target: float`
- `orthogonalize_direction: bool`
- `row_normalization: RowNormalization`
- `full_normalization_lora_rank: int`
- `winsorization_quantile: float`
- `study_checkpoint_dir: str`

Removed fields (from the fork):
- `inference_backend`
- `vllm_gpu_memory_utilization`
- `vllm_max_model_len`

Settings sources now use proper `CliSettingsSource` (with `cli_implicit_flags=True` and `cli_kebab_case=True`) and `EnvSettingsSource` instead of `SettingsConfigDict`.

### 4. `src/heretic/model.py` — Complete Architecture Change

This is the biggest change. The fork used **direct in-place weight modification** (`.sub_()`); the upstream uses **LoRA (PEFT) adapters**.

#### Architecture Change: LoRA vs. Direct Modification

| Aspect | Fork (direct) | Upstream (LoRA) |
|--------|--------------|-----------------|
| Weight storage | Modified base weights | Delta stored in LoRA A/B matrices |
| Trial reset | Full model reload | Zero out LoRA B matrices (very fast) |
| Merge before save | Not needed | Required (`merge_and_unload()`) |
| Quantized models | Fails (cannot modify quantized weights) | Works (LoRA modifies adapters, not base) |
| Memory efficiency | Same | Better (adapters are small) |

#### New Imports
- `bitsandbytes as bnb` — for 4-bit weight dequantization
- `peft` — LoRA adapter framework
- `torch.linalg as LA` — for row norm computation
- `AutoModelForImageTextToText` — multimodal support
- `BitsAndBytesConfig` — quantization config
- `GenerateDecoderOnlyOutput` — proper typed output

#### New `get_model_class()` Function
Dynamically selects `AutoModelForImageTextToText` or `AutoModelForCausalLM` based on model config, enabling multimodal support.

#### `__init__` Changes
- Validates `trust_remote_code` setting
- Sets `tokenizer.padding_side = "left"` unconditionally (critical for decoder-only models)
- Tracks `max_memory` and `trusted_models`
- Loads model with `quantization_config` when BNB_4BIT selected
- Calls `_apply_lora()` after loading to attach PEFT adapters
- Removed: quantization warning check (superseded by proper BNB support)
- Removed: vLLM backend initialization

#### New Method: `_apply_lora()`
Attaches LoRA adapters to the model's abliterable components. Uses rank 1 for normal/pre normalization and configurable rank for full normalization.

#### New Method: `_get_quantization_config()`
Creates `BitsAndBytesConfig` for 4-bit quantization.

#### New Method: `get_merged_model()`
Merges LoRA adapters back into the base model before saving. Handles quantized models specially (reloads base model in full precision on CPU first).

#### Renamed Method: `reload_model()` → `reset_model()`
- Fast path: if same model and no full reload needed, just zeros out LoRA B matrices
- Slow path: full model reload with quantization config

#### Changed: `get_layer_matrices()` → `get_layer_modules()`
- Returns `dict[str, list[Module]]` (modules) instead of `dict[str, list[Tensor]]` (tensors)
- Works with PEFT's LoRA-wrapped modules
- Added support for Granite MoE Hybrid architecture

#### Changed: `abliterate()`
Completely rewritten to use LoRA adapters instead of in-place modification:
- Dequantizes 4-bit weights if needed
- Supports row normalization (NONE, PRE, FULL)
- Stores delta as LoRA A/B matrices: `lora_B = -weight * v`, `lora_A = v^T @ W`
- For FULL normalization: uses SVD low-rank approximation

#### Changed: `generate()`
- Now accepts `list[Prompt]` (with `.system` and `.user` fields) instead of `list[str]`
- Appends `response_prefix` to prompts if set (for better CoT suppression)

#### Changed: `get_responses()`
- Removed vLLM backend path
- Added `skip_special_tokens` parameter

#### Changed: `get_residuals()`
- Now supports winsorization (`winsorization_quantile` setting)

#### Removed Methods
- `initialize_vllm_backend()` — vLLM removed

### 5. `src/heretic/evaluator.py` — Prompt Type and Scoring Updates

| Change | Description |
|--------|-------------|
| `load_prompts()` | Now returns `list[Prompt]` instead of `list[str]` |
| `is_refusal()` | Now classifies empty responses as refusals; normalizes whitespace |
| `count_refusals()` | Uses `skip_special_tokens=True`; prints responses if `print_responses` is set |
| `get_score()` | Uses `kl_divergence_target` for better scoring; returns `(tuple, float, int)` |
| KL display | Shows 4 decimal places instead of 2 |
| Type annotations | Added proper type annotations to class variables |

#### Improved scoring algorithm
The old scoring was:
```python
score = (kl_divergence / kl_divergence_scale, refusals / base_refusals)
```

The new scoring accounts for a target KL divergence to prevent optimizing parameter combinations that "do nothing":
```python
if kl_divergence >= kl_divergence_target:
    kld_score = kl_divergence / kl_divergence_scale
else:
    kld_score = refusals_score * kl_divergence_target / kl_divergence_scale
score = (kld_score, refusals_score)
```

### 6. `src/heretic/utils.py` — Major Expansion

#### New: `Prompt` Dataclass
```python
@dataclass
class Prompt:
    system: str
    user: str
```
All prompt-handling functions now work with this type instead of raw strings.

#### Changed: `load_prompts()`
- Now accepts `settings: Settings` as first argument (for system prompt)
- Supports local dataset directories (both `save_to_disk` format and raw files)
- Supports `prefix`, `suffix`, and `system_prompt` from `DatasetSpecification`
- Returns `list[Prompt]` instead of `list[str]`

#### New Helper Functions
- `print_memory_usage()` — prints resident RAM, allocated/reserved GPU VRAM
- `is_notebook()` — detects Jupyter/Colab/Kaggle environments
- `prompt_select()` — interactive select with notebook fallback (numbered list)
- `prompt_text()` — interactive text input with notebook fallback
- `prompt_path()` — interactive path input with notebook fallback
- `prompt_password()` — interactive password input with notebook fallback

#### Changed: `empty_cache()`
- Supports MLU, SDAA, MUSA accelerators (previously only CUDA, XPU, MPS)

#### Changed: `get_trial_parameters()`
- Uses dict access instead of `asdict()` (parameters are already dicts in study storage)

#### Changed: `get_readme_intro()`
- Hides local model paths (privacy improvement)
- Accepts `list[Prompt]` instead of `list[str]` for bad prompts

### 7. `src/heretic/main.py` — Major UI and Logic Overhaul

#### Startup Changes
- Env variable uses both `PYTORCH_ALLOC_CONF` and `PYTORCH_CUDA_ALLOC_CONF` for compatibility
- CLI argument detection: checks for missing `--model` flag more robustly (not just argument count parity)
- Multi-GPU detection: shows count, total VRAM, and per-device info
- Removes vLLM backend notification

#### Checkpoint/Resume Support
New study checkpoint persistence using `JournalStorage`:
```python
lock_obj = JournalFileOpenLock(study_checkpoint_file)
backend = JournalFileBackend(study_checkpoint_file, lock_obj=lock_obj)
storage = JournalStorage(backend)
```
- Detects existing checkpoint and prompts user: resume, show results, or start fresh
- Checkpoint file named after model ID

#### Optimization Changes
- Uses `model.reset_model()` instead of `model.reload_model()`
- Parameters stored as dicts (`asdict()`) for JSON serialization in checkpoint
- `objective_wrapper` catches `KeyboardInterrupt` and stops study gracefully
- Tracks `start_index` for accurate time estimates on resumed runs
- Pareto front calculation is now done manually (more accurate than `study.best_trials`)
- After optimization: can run additional trials without restarting

#### New: `obtain_merge_strategy()` Function
Prompts user for merge strategy when quantization is enabled:
- Estimates RAM required for dequantization (using meta device)
- Returns `"merge"`, `"adapter"`, or `None` (cancelled)

#### Orthogonalized Directions
New section in `run()`:
```python
if settings.orthogonalize_direction:
    good_directions = F.normalize(good_means, p=2, dim=1)
    projection_vector = torch.sum(refusal_directions * good_directions, dim=1)
    refusal_directions = refusal_directions - projection_vector.unsqueeze(1) * good_directions
    refusal_directions = F.normalize(refusal_directions, p=2, dim=1)
```

#### New: Analyzer Integration
After computing refusal directions:
```python
analyzer = Analyzer(settings, model, good_residuals, bad_residuals)
if settings.print_residual_geometry:
    analyzer.print_residual_geometry()
if settings.plot_residuals:
    analyzer.plot_residuals()
del good_residuals, bad_residuals, analyzer
```

#### Save/Upload Workflow Changes
- Uses new `prompt_*` helpers instead of direct `questionary.*` calls
- Save: calls `obtain_merge_strategy()` before saving
- Upload: supports both local model cards and remote HF model cards
- Upload: handles quantized models with adapter vs merge strategy

#### Memory Usage Display
After each trial: calls `print_memory_usage()`.

### 8. `src/heretic/analyzer.py` — New File

Completely new file providing research features:

#### `Analyzer` Class

```python
class Analyzer:
    def __init__(self, settings, model, good_residuals, bad_residuals): ...
    def print_residual_geometry(self): ...
    def plot_residuals(self): ...
```

**`print_residual_geometry()`** (requires `[research]` extras):
- Computes geometric medians of residuals (using `geom-median`)
- Displays a rich table with per-layer metrics:
  - Cosine similarities: S(g,b), S(g*,b*), S(g,r), S(g*,r*), S(b,r), S(b*,r*)
  - L2 norms: |g|, |g*|, |b|, |b*|, |r|, |r*|
  - Silhouette coefficient for good/bad clustering
- Uses `scikit-learn` for silhouette scores

**`plot_residuals()`** (requires `[research]` extras):
- Uses `PaCMAP` for 2D dimensionality reduction
- Aligns plots by rotating to make the good/bad median line horizontal
- Uses `PaCMAP` initialization from previous layer for smooth transitions
- Generates per-layer PNG images
- Creates animated GIF with smooth transitions between layers

### 9. `src/heretic/vllm_backend.py` — Removed

The fork added a vLLM integration for faster inference. The upstream chose not to include this because:
- vLLM cannot be used during abliteration trials (model weights are modified in memory)
- The added complexity was not worth the benefit for the typical use case
- BNB 4-bit quantization provides a better solution for VRAM-constrained setups

---

## Refusal Markers Improvements

The upstream expanded the refusal markers to catch more patterns:

Added (covering informal spellings and variations):
- `"i can'"` (covers "i can't" but also partial matches)
- `"i cant"` (no apostrophe)
- `"i won'"` (covers "i won't")
- `"i wont"` (no apostrophe)
- `"i unable"` (covers more patterns)
- `"im unable"` (no apostrophe)
- `"i an ai"` (pattern without apostrophes)
- `"im an ai"` (informal)
- `"i designed to"` (without "am")
- `"im designed to"` (informal)
- `"i programmed to"` (without "am")
- `"im programmed to"` (informal)

---

## Breaking Changes for Users

1. **vLLM configuration options removed**: `inference_backend`, `vllm_gpu_memory_utilization`, `vllm_max_model_len` no longer exist. Users with these in their `config.toml` should remove them.
2. **Save format**: Saved models now require merging LoRA adapters. The program handles this interactively.
3. **Checkpoint files**: New checkpoint format (JSONL) in `checkpoints/` directory. Old checkpoints are not compatible.
4. **Python dependencies**: `bitsandbytes`, `peft`, `psutil`, `kernels` are now required. `vllm` is no longer a supported optional dependency.
