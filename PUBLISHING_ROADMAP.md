# Publishing Roadmap — napari-organelle-contact-analyzer

Goal: take the plugin from its current working state to a published GitHub release listed on napari hub.

## Where things stand today

The plugin is further along than a typical "early stage" project. `_widget.py` is a full-featured, ~2,700-line implementation (`OrganelleContactWidget`) covering thresholding, ROI handling, contact metrics, scale bars, and Excel/GraphPad export. The repo already has a napari plugin manifest, GitHub Actions CI/CD (including a PyPI deploy job), pre-commit hooks, and `pyproject.toml` metadata. What's missing is the cleanup, testing, and documentation layer between "works for me" and "safe to publish."

## Phase 1 — Code cleanup

- [ ] Delete `thresholding.py` (or repurpose it — your call). Its only function is superseded by the per-channel thresholding already in `_widget.py`, and it currently launches a napari viewer as a side effect of being imported.
- [ ] Review the monkey-patches at the top of `_widget.py` (tifffile `RESUNIT` attribute, napari's `polygon_creating`). Both are wrapped in bare `except Exception: pass`, so failures fail silently. Decide whether to keep them, narrow the exception handling, or replace with proper version pinning.
- [ ] Decide the fate of `_reader.py`, `_writer.py`, `_sample_data.py` — still unedited cookiecutter template stubs (reader only handles `.npy`, writer is a no-op, sample data is random noise). Either implement them for real or remove their entries from `napari.yaml` so the manifest doesn't advertise non-functional features.
- [ ] Fix the unfilled template placeholder in `tox.ini`: `--cov={{module_name}}` → `--cov=napari_organelle_contact_analyzer`.
- [ ] Commit or discard the currently staged/unstaged changes to `pyproject.toml`, `_widget.py`, `__init__.py`, `napari.yaml`.
- [ ] Remove the stray untracked file `napari.yaml copy old`.

## Phase 2 — Tests & CI

- [ ] Rewrite `_tests/test_widget.py` — it currently imports `ExampleQWidget`, `ImageThreshold`, `threshold_autogenerate_widget`, `threshold_magic_widget`, none of which exist anymore. Tests fail at collection.
- [ ] Add unit tests for the pure helper functions in `_widget.py` (`compute_contact_density`, `compute_roi_geometry`, `compute_additional_metrics`, `safe_mean_intensity`, `parse_channel_list_text`) — these don't need a running viewer and are the easiest, highest-value tests to add.
- [ ] Add at least a smoke test that instantiates `OrganelleContactWidget` with `make_napari_viewer` and confirms it builds without error.
- [ ] Run `tox` locally (or `pytest` directly) until green, then confirm the GitHub Actions workflow passes on a pushed branch.
- [ ] Run pre-commit (`ruff`, `black`, `napari-plugin-checks`) and resolve any flagged issues.

## Phase 3 — Documentation

- [ ] Rewrite `README.md`: remove the copier template boilerplate and the stray "Yay it worked!" line. Add a real description, installation instructions, a usage walkthrough (channel numbering, thresholding modes, output metrics), and ideally a screenshot or short GIF of the widget in use.
- [ ] Write `.napari-hub/DESCRIPTION.md` — this is what actually renders on the napari hub plugin page, separate from the README, and should be written for the end-user audience (microscopists/cell biologists), not developers.
- [ ] Fill in `.napari-hub/config-yml` if you want hub-specific overrides (summary line, etc.).
- [ ] Confirm `LICENSE` and license classifier are correct (currently MPL 2.0 — already set).

## Phase 4 — Packaging & versioning

- [ ] Update the `Development Status` classifier in `pyproject.toml` (currently `2 - Pre-Alpha`) to reflect actual maturity before release (e.g. `3 - Alpha` or `4 - Beta`).
- [ ] Tag a version, e.g. `v0.1.0` — `setuptools_scm` derives the package version from git tags, and none exist yet.
- [ ] Build locally (`python -m build`) and sanity-check the resulting wheel/sdist.
- [ ] Validate the plugin manifest (`napari.yaml`) with `npe2 validate .` or `python -m npe2 validate`.

## Phase 5 — Publish to PyPI

- [ ] Create a PyPI account and API token if you don't already have one.
- [ ] Add the token as the `TWINE_API_KEY` secret in the GitHub repo settings — the existing CI workflow already builds and uploads via Twine when a `v*` tag is pushed.
- [ ] Push the version tag to trigger the release job.
- [ ] Verify the package installs cleanly in a fresh environment: `pip install napari-organelle-contact-analyzer`.

## Phase 6 — Napari hub listing

- [ ] napari hub auto-indexes any PyPI package carrying the `Framework :: napari` trove classifier (already present in `pyproject.toml`) — no separate submission form, but it can take up to ~24h to appear after a PyPI release.
- [ ] Check the live listing at napari-hub.org once it appears; confirm `DESCRIPTION.md` and metadata render as expected.
- [ ] Iterate on description/screenshots based on how the listing actually looks.

## Suggested order

Phases 1 → 2 → 3 can happen roughly in that order, but Phase 2 (tests passing) is really the gate that tells you Phase 1's cleanup didn't break anything — worth treating as a checkpoint before moving to docs and packaging. Phases 4–6 are strictly sequential (can't publish to PyPI without a tag, can't list on the hub without a PyPI release).
