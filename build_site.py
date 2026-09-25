#!/usr/bin/env python3
"""Wrap app.html into a full page and copy the latest data into _site/ for GitHub Pages."""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "_site")

HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#EEF1F4" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#0E151B" media="(prefers-color-scheme: dark)">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="Edge Desk">
<link rel="manifest" href="manifest.webmanifest">
<link rel="icon" href="icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="icon.svg">
<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}
body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
"""

ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><rect width="64" height="64" rx="14" fill="#132029"/>
<path d="M12 44l12-14 9 7 17-21" fill="none" stroke="#F08A45" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/></svg>"""

MANIFEST = """{"name":"Edge Desk","short_name":"Edge Desk","start_url":"./","display":"standalone",
"background_color":"#0E151B","theme_color":"#132029","icons":[{"src":"icon.svg","sizes":"any","type":"image/svg+xml"}]}"""


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "data"))
    body = open(os.path.join(ROOT, "app.html"), encoding="utf-8").read()
    # app.html starts with <title>, <link>s and <style>; they are valid inside <head>.
    split = body.index("</style>") + len("</style>")
    page = HEAD + body[:split] + "\n</head>\n<body>\n" + body[split:] + "\n</body>\n</html>\n"
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(page)
    open(os.path.join(OUT, "icon.svg"), "w").write(ICON)
    open(os.path.join(OUT, "manifest.webmanifest"), "w").write(MANIFEST)
    for f in ("feed.json", "paper.json", "stats.json"):
        src = os.path.join(ROOT, "data", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT, "data", f))
    open(os.path.join(OUT, ".nojekyll"), "w").close()
    print("built", OUT)


if __name__ == "__main__":
    main()
