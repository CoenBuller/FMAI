from PIL import Image

def gif_contact_sheet(gif_path, out_path, n_frames=12, cols=4):
    im = Image.open(gif_path)
    total = im.n_frames
    idx = [round(i * (total - 1) / (n_frames - 1)) for i in range(n_frames)]
    frames = []
    for i in idx:
        im.seek(i)
        frames.append(im.convert("RGB").copy())
    w, h = frames[0].size
    rows = -(-n_frames // cols)
    sheet = Image.new("RGB", (cols * w, rows * h), "white")
    for k, f in enumerate(frames):
        sheet.paste(f, ((k % cols) * w, (k // cols) * h))
    sheet.save(out_path)

gif_contact_sheet("results/second/gifs/20261030_134705/navigation_none_seed67.gif", "sheet.png")