# minimal_wizard.py
from __future__ import annotations
import json, re, getpass
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Tuple

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box
from PIL import Image

app = typer.Typer(add_completion=False)
con = Console()

# Data model 
@dataclass
class Basics:
    image_path: str
    mode: str                  # "2d" | "3d"
    out_dir: str
    out_name: str

@dataclass
class Provider:
    name: str                  # "a1111" | "replicate" | "meshy"
    api_key: Optional[str]     # masked; not persisted unless proceed

@dataclass
class Output:
    size: Tuple[int, int]      # (W, H)
    batch: int

@dataclass
class Style2D:
    style: str                 # "photoreal" | "cartoon" | "watercolor" | "sketch"
    strictness: str            # "low" | "medium" | "high"

@dataclass
class Style3D:
    quality: str               # "draft" | "standard" | "high"
    style: str                 # "real" | "toy" | "sculpture"

@dataclass
class Session:
    basics: Basics
    provider: Provider
    output: Output
    object_type: str           # "creature" | "human" | "vehicle" | "building" | "prop" | "abstract"
    style2d: Optional[Style2D]
    style3d: Optional[Style3D]

# UI helpers
def section(title: str, sub: str = ""):
    txt = f"[b]{title}[/b]" + (f"\n[subdued]{sub}[/subdued]" if sub else "")
    con.print(Panel.fit(txt, border_style="cyan"))

def ask_image() -> str:
    while True:
        p = Prompt.ask("Path (PNG/JPG)")
        path = Path(p).expanduser()
        if not path.exists():
            con.print("[red]File not found.[/red]"); continue
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            con.print("[red]Use PNG/JPG.[/red]"); continue
        try:
            with Image.open(path) as im:
                w, h = im.size
            con.print(f"[green]✓ Loaded {w}×{h}[/green]")
            return str(path)
        except Exception:
            con.print("[red]Could not open image.[/red]")

def ask_mode() -> str:
    con.print("[1] 2D render   [2] 3D model")
    c = Prompt.ask("Choose 1 or 2", choices=["1","2"])
    return "2d" if c == "1" else "3d"

def ask_output_basics(image_path: str, mode: str) -> Tuple[str, str]:
    con.print("[dim](Press Enter to use the current folder for output)[/dim]")
    out_dir = Prompt.ask("Output folder", default=".")
    stem = Path(image_path).stem
    default_name = f"{stem}_{mode}"
    out_name = Prompt.ask("Output filename (without extension)", default=default_name)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return out_dir, out_name

def ask_provider(mode: str) -> Provider:
    con.print("(1) Local A1111 (2D)   (2) Replicate (2D)   (3) Meshy (3D)")
    valid = {"1":"a1111","2":"replicate","3":"meshy"}
    while True:
        pick = Prompt.ask("Provider 1/2/3", choices=["1","2","3"])
        name = valid[pick]
        if mode == "3d" and name != "meshy":
            con.print("[yellow]Tip:[/yellow] For 3D, choose Meshy (3).")
        if mode == "2d" and name == "meshy":
            con.print("[yellow]Note:[/yellow] Meshy is 3D only.")
        break
    api_key = None
    if name in {"replicate","meshy"}:
        con.print("[dim]API key will be hidden and not written to disk unless you proceed.[/dim]")
        api_key = getpass.getpass("API key: ")
    return Provider(name=name, api_key=api_key or None)

def ask_size_batch() -> Tuple[Tuple[int,int], int]:
    con.print("Size: (a)512 (b)768 (c)1024 (d)1536 (e)Custom WxH")
    while True:
        pick = Prompt.ask("Choose a/b/c/d/e", choices=list("abcde"))
        if pick == "a": size = (512,512)
        elif pick == "b": size = (768,768)
        elif pick == "c": size = (1024,1024)
        elif pick == "d": size = (1536,1536)
        else:
            raw = Prompt.ask("Enter WxH (e.g., 960x1280)")
            m = re.match(r"^\s*(\d+)\s*[xX]\s*(\d+)\s*$", raw)
            if not m: con.print("[red]Format WxH required.[/red]"); continue
            size = (int(m.group(1)), int(m.group(2)))
        break
    while True:
        b = Prompt.ask("Batch count [1–8]", default="1")
        try:
            batch = int(b)
            if 1 <= batch <= 8: break
        except: pass
        con.print("[red]Enter an integer 1–8.[/red]")
    return size, batch

def ask_object_type() -> str:
    con.print("Object type: (1) Creature (2) Human (3) Vehicle (4) Building (5) Prop (6) Abstract")
    m = {"1":"creature","2":"human","3":"vehicle","4":"building","5":"prop","6":"abstract"}
    pick = Prompt.ask("Choose 1–6", choices=list(m.keys()))
    return m[pick]

def ask_style_2d() -> Style2D:
    con.print("2D Style: (1) Photoreal (2) Cartoon (3) Watercolor (4) Sketch")
    s_map = {"1":"photoreal","2":"cartoon","3":"watercolor","4":"sketch"}
    style = s_map[Prompt.ask("Choose 1–4", choices=list(s_map.keys()))]
    con.print("Control strictness: (1) Low (2) Medium (3) High")
    st_map = {"1":"low","2":"medium","3":"high"}
    strict = st_map[Prompt.ask("Choose 1–3", choices=list(st_map.keys()))]
    return Style2D(style=style, strictness=strict)

def ask_style_3d() -> Style3D:
    con.print("3D Quality: (1) Draft (2) Standard (3) High")
    q_map = {"1":"draft","2":"standard","3":"high"}
    quality = q_map[Prompt.ask("Choose 1–3", choices=list(q_map.keys()))]
    con.print("3D Style: (1) Realistic (2) Toy/Cartoon (3) Sculpture")
    s_map = {"1":"realistic","2":"toy","3":"sculpture"}
    style = s_map[Prompt.ask("Choose 1–3", choices=list(s_map.keys()))]
    return Style3D(quality=quality, style=style)

def summary(sess: Session):
    tbl = Table(title="Session Summary", box=box.SIMPLE)
    tbl.add_column("Field", style="cyan", no_wrap=True)
    tbl.add_column("Value", style="white")
    b = sess.basics
    o = sess.output
    p = sess.provider
    tbl.add_row("Mode", b.mode.upper())
    tbl.add_row("Image", b.image_path)
    tbl.add_row("Output", f"{Path(b.out_dir, b.out_name)}.[ext]")
    tbl.add_row("Size / Batch", f"{o.size[0]}×{o.size[1]} / {o.batch}")
    tbl.add_row("Provider", p.name + (" (key set)" if p.api_key else ""))
    tbl.add_row("Object type", sess.object_type)
    if sess.style2d:
        tbl.add_row("2D • Style", sess.style2d.style)
        tbl.add_row("2D • Strictness", sess.style2d.strictness)
    if sess.style3d:
        tbl.add_row("3D • Quality", sess.style3d.quality)
        tbl.add_row("3D • Style", sess.style3d.style)
    con.print(Panel(tbl, border_style="green"))

# Main
@app.command(help="Collect minimal settings for 2D/3D generation (no model calls).")
def run():
    section("GenAI@Berkeley <> Netflix Image Pipeline")

    # 1) Input
    section("Step 1 — Input image")
    img = ask_image()

    # 2) Mode
    section("Step 2 — Mode")
    mode = ask_mode()

    # 3) Output
    section("Step 3 — Output")
    out_dir, out_name = ask_output_basics(img, mode)

    # 4) Provider
    section("Step 4 — Provider")
    provider = ask_provider(mode)

    # 5) Size + Batch
    section("Step 5 — Output size & batch")
    size, batch = ask_size_batch()

    # 6) Object type
    section("Step 6 — Object type")
    obj_type = ask_object_type()

    # 7) Style basics
    style2d = None
    style3d = None
    if mode == "2d":
        section("Step 7 — 2D style")
        style2d = ask_style_2d()
    else:
        section("Step 7 — 3D style")
        style3d = ask_style_3d()

    sess = Session(
        basics=Basics(image_path=img, mode=mode, out_dir=out_dir, out_name=out_name),
        provider=Provider(name=provider.name, api_key=provider.api_key),
        output=Output(size=size, batch=batch),
        object_type=obj_type,
        style2d=style2d,
        style3d=style3d
    )

    section("Review & Confirm")
    summary(sess)

    if not Confirm.ask("Proceed? (Stub only — no model calls yet)", default=True):
        con.print("[yellow]Cancelled.[/yellow]")
        raise typer.Exit()

    # Save minimal session snapshot (no API keys persisted unless you want)
    snap = asdict(sess)
    # Remove the API key from the snapshot by default for safety:
    snap["provider"]["api_key"] = None
    with open(".last_session_min.json", "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)

    con.print("[green]✓ Saved session → ./.last_session_min.json[/green]")
    con.print("[dim](Todo: add the real model calls)[/dim]")

if __name__ == "__main__":
    app()
