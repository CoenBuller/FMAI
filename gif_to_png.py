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

if __name__ == "__main__":
    import sys
    gif_contact_sheet(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "sheet.png")   # python gif_to_png.py <gif> [out.png]