"""19 image questions with unambiguous one-word answers, shared by both paths."""
from PIL import Image, ImageDraw

COLOURS = {
    "red": (255, 0, 0), "blue": (0, 0, 255), "green": (0, 160, 0), "yellow": (255, 255, 0),
    "black": (0, 0, 0), "white": (255, 255, 255), "orange": (255, 140, 0), "purple": (128, 0, 128),
}
SIZE = 256

def solid(c):
    return Image.new("RGB", (SIZE, SIZE), COLOURS[c])

def split(left, right):
    im = Image.new("RGB", (SIZE, SIZE), COLOURS[left])
    im.paste(Image.new("RGB", (SIZE // 2, SIZE), COLOURS[right]), (SIZE // 2, 0, SIZE, SIZE))
    return im

def shape(kind, colour="black"):
    im = Image.new("RGB", (SIZE, SIZE), (255, 255, 255))
    d = ImageDraw.Draw(im)
    box = (48, 48, SIZE - 48, SIZE - 48)
    if kind == "circle":
        d.ellipse(box, fill=COLOURS[colour])
    elif kind == "square":
        d.rectangle(box, fill=COLOURS[colour])
    elif kind == "triangle":
        d.polygon([(SIZE // 2, 48), (48, SIZE - 48), (SIZE - 48, SIZE - 48)], fill=COLOURS[colour])
    return im

Q_COLOUR = "What colour is this image? Answer with one word."
Q_LEFT = "Which colour is on the left half? Answer with one word."
Q_SHAPE = "What shape is shown? Answer with one word."

def cases():
    out = []
    for c in ("red", "blue", "green", "yellow", "black", "white", "orange", "purple"):
        out.append((f"solid-{c}", solid(c), Q_COLOUR, c))
    for left, right in (("red", "blue"), ("blue", "red"), ("green", "red"), ("red", "green"),
                        ("blue", "green"), ("green", "blue"), ("yellow", "blue"), ("black", "white")):
        out.append((f"split-{left}-{right}", split(left, right), Q_LEFT, left))
    for s in ("circle", "square", "triangle"):
        out.append((f"shape-{s}", shape(s), Q_SHAPE, s))
    return out
