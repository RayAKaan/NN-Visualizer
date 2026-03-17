import { useRef, useState } from "react";
import { useLabStore } from "../../store/labStore";

export function ImageSelector() {
  const setInputImage = useLabStore((s) => s.setInputImage);
  const [preview, setPreview] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const processImage = async (src: string) => {
    const img = new Image();
    img.src = src;
    await new Promise((resolve) => {
      img.onload = resolve;
    });

    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(img, 0, 0, 64, 64);
    const data = ctx.getImageData(0, 0, 64, 64).data;
    const pixels = new Float32Array(3 * 64 * 64);
    for (let i = 0; i < 64 * 64; i += 1) {
      pixels[i] = data[i * 4] / 255;
      pixels[i + 64 * 64] = data[i * 4 + 1] / 255;
      pixels[i + 2 * 64 * 64] = data[i * 4 + 2] / 255;
    }

    setPreview(src);
    setInputImage(src, pixels);
  };

  return (
    <div className="space-y-3">
      <button
        type="button"
        onClick={() => fileRef.current?.click()}
        className="grid h-[280px] w-[280px] place-items-center rounded-2xl border border-dashed border-slate-600 bg-slate-900/40 text-center text-xs text-slate-400 transition hover:border-cyan-400/55"
      >
        {preview ? (
          <img src={preview} alt="Selected" className="h-full w-full rounded-2xl object-cover" />
        ) : (
          <span>Click to upload cat/dog image</span>
        )}
      </button>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (!file) return;
          const url = URL.createObjectURL(file);
          void processImage(url);
        }}
      />
    </div>
  );
}
