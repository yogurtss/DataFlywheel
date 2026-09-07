"""Create a small, explicitly synthetic local dataset; no model required."""
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/demo-input")
    root.mkdir(parents=True, exist_ok=True)
    labels = [("table", "<table><tr><td>Year</td><td>Revenue</td></tr><tr><td>2025</td><td>100</td></tr></table>", "html"),
              ("text", "The quick brown fox jumps over the lazy dog.", "text"),
              ("formula", "x^{2}+y^{2}=z^{2}", "latex")]
    rows = []
    for i, (task, gt, fmt) in enumerate(labels):
        image = Image.new("RGB", (700, 140), "white")
        draw = ImageDraw.Draw(image)
        if task == "table":
            for x in (20, 250, 500):
                draw.line((x, 15, x, 115), fill="black", width=2)
            for y in (15, 65, 115):
                draw.line((20, y, 500, y), fill="black", width=2)
            for x, y, text in [(30,30,"Year"),(260,30,"Revenue"),(30,80,"2025"),(260,80,"100")]:
                draw.text((x,y), text, fill="black", font_size=22)
        else:
            draw.text((20, 45), "x² + y² = z²" if task == "formula" else gt, fill="black", font_size=22)
        image.save(root / f"{task}.png")
        rows.append({"id": task, "task": task, "image": f"{task}.png", "gt": gt, "gt_format": fmt,
                     "source": "synthetic_demo", "domain": "private", "document_id": f"demo-{i}", "split": "train"})
    (root / "input.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(root / "input.json")


if __name__ == "__main__":
    main()
