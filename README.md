# concertyy

Personal blog at <https://x2yzen.github.io> — *an orchestra of random thoughts*. Built on Jekyll with the [Chirpy](https://github.com/cotes2020/jekyll-theme-chirpy) theme (consumed as a gem) and deployed to GitHub Pages by `.github/workflows/pages-deploy.yml` on every push to `main`.

## Develop

```bash
bundle install            # one-time setup (needs Ruby 3.1+)
bash tools/run.sh         # dev server with livereload → 127.0.0.1:4000
bash tools/run.sh -p      # serve in JEKYLL_ENV=production
bash tools/test.sh        # production build + html-proofer (matches CI)
```

Run `tools/test.sh` before pushing if a post adds links or images.

## Layout

```
_posts/                       # blog posts (YYYY-MM-DD-slug.md)
_tabs/                        # top-level pages (About, Archives, …)
_data/contact.yml             # sidebar contact icons
_config.yml                   # site title, SEO, plugins, …
assets/images/<post-slug>/    # images for each post
assets/img/avatar.png         # sidebar avatar
vitrine/                      # standalone interactive HTML pieces
textures/                     # HDR/PBR assets used by /vitrine/
```

Theme files (`_layouts`, `_includes`, `_sass`) live inside the
`jekyll-theme-chirpy` gem — locate them with `bundle info --path jekyll-theme-chirpy`. Customize by overriding locally or by bumping the gem in `Gemfile`.

## Authoring

See [CLAUDE.md](./CLAUDE.md) for the frontmatter template, asset directory convention, and the one timezone rule that matters: a post must not be future-dated relative to site time at build, or Jekyll silently drops it.

## License

[MIT](./LICENSE).
