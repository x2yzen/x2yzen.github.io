# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A personal Jekyll blog built on the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme (consumed as a gem, not vendored). Deployed to GitHub Pages from `main` via `.github/workflows/pages-deploy.yml`. Almost all changes here are content (posts), not theme code — the theme lives inside the `jekyll-theme-chirpy` gem (see `bundle info --path jekyll-theme-chirpy`) and should not be modified locally.

## Common commands

```bash
bundle install              # one-time: install Jekyll + theme + html-proofer
bash tools/run.sh           # local dev server with livereload (127.0.0.1:4000)
bash tools/run.sh -p        # serve in JEKYLL_ENV=production (closer to deployed output)
bash tools/test.sh          # production build + html-proofer (matches CI)
```

`tools/test.sh` is what CI runs — use it before pushing if a post adds links or images. It cleans `_site/` and runs htmlproofer with external URL checks disabled.

## Authoring posts

Posts live in `_posts/` with filename `YYYY-MM-DD-slug.md`. The slug becomes the URL because `_config.yml` sets `permalink: /posts/:title/`.

Required front matter (copy from any existing post):

```yaml
---
title: <Title>
date: YYYY-MM-DD HH:MM:SS -0700      # site tz is America/Los_Angeles; write dates in Pacific (-0700 PDT / -0800 PST). Future-dated posts (relative to site tz at build time) are silently dropped by Jekyll.
categories: [<one_category>]          # singular; existing values: language_model, statistics, game, reinforcement_learning
tags: [<tag1>, <tag2>]
pin: false
math: true|false                      # set true to load MathJax for this post
---
```

Post images go in `assets/images/<post-slug>/` and are referenced as `/assets/images/<post-slug>/<file>`. Keep the directory name in sync with the post filename's slug.

`last_modified_at` is set automatically by `_plugins/posts-lastmod-hook.rb` from `git log` — do not set it manually. This means the displayed "last modified" only updates after the change is committed.

## Theme customization gotcha

`_config.yml`, `_plugins/`, `_tabs/`, and `index.html` are copied from the Chirpy starter so the gem-based install works end-to-end (Jekyll only reads `_data`, `_layouts`, `_includes`, `_sass`, `assets` from the theme gem). Editing layouts/styles requires either overriding by creating matching files locally, or upgrading the gem version in `Gemfile`.

## Submodule

`assets/lib` is a git submodule pointing at `chirpy-static-assets`. It is **not** initialized by default and the deploy workflow does not check it out (see the commented-out `submodules: true` in the workflow). Don't init it unless you intend to switch to self-hosted assets (which also requires flipping `assets.self_host.enabled` in `_config.yml`).
