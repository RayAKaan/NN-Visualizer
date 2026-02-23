import React, { useEffect, useRef } from "react";

interface Props {
  pixels: number[];
}

const InputHeatmap = React.memo(function InputHeatmap({ pixels }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i += 1) {
      const v = Math.round(Math.max(0, Math.min(1, pixels[i] ?? 0)) * 255);
      const idx = i * 4;
      img.data[idx] = v;
      img.data[idx + 1] = v;
      img.data[idx + 2] = v;
      img.data[idx + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels]);

  return (
    <div className="layer-block" data-highlight="inputLayer">
      <div className="input-heatmap">
        <canvas
          ref={canvasRef}
          width={28}
          height={28}
          style={{ width: 84, height: 84, imageRendering: "pixelated" }}
        />
      </div>
      <span className="layer-label">Input (784)</span>
    </div>
  );
});

export default InputHeatmap;
