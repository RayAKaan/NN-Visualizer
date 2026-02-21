import React, { useEffect, useRef, useState } from "react";

interface Props {
  pixels: number[];
  onChange: (pixels: number[]) => void;
  brushSize: 1 | 2 | 3;
}

const SIZE = 28;
const SCALE = 10;

const applyBrush = (grid: number[], cx: number, cy: number, brushSize: 1 | 2 | 3, erase: boolean) => {
  const radius = brushSize - 1;
  for (let y = cy - radius; y <= cy + radius; y += 1) {
    for (let x = cx - radius; x <= cx + radius; x += 1) {
      if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) continue;
      const d = Math.hypot(x - cx, y - cy);
      const falloff = Math.max(0, 1 - d / (radius + 1));
      const idx = y * SIZE + x;
      const value = erase ? 0 : Math.max(grid[idx], falloff);
      grid[idx] = value;
    }
  }
};

const CanvasGrid: React.FC<Props> = ({ pixels, onChange, brushSize }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [painting, setPainting] = useState(false);
  const [erase, setErase] = useState(false);
  const [hover, setHover] = useState<[number, number] | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#111827";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < SIZE; y += 1) {
      for (let x = 0; x < SIZE; x += 1) {
        const v = pixels[y * SIZE + x] ?? 0;
        if (v > 0) {
          ctx.fillStyle = `rgba(255,255,255,${Math.min(1, v + 0.1)})`;
          ctx.fillRect(x * SCALE, y * SCALE, SCALE, SCALE);
        }
      }
    }

    ctx.strokeStyle = "#1a2035";
    for (let i = 0; i <= SIZE; i += 1) {
      ctx.beginPath();
      ctx.moveTo(i * SCALE, 0);
      ctx.lineTo(i * SCALE, SIZE * SCALE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * SCALE);
      ctx.lineTo(SIZE * SCALE, i * SCALE);
      ctx.stroke();
    }

    if (hover) {
      ctx.strokeStyle = "rgba(6,182,212,0.6)";
      ctx.strokeRect(hover[0] * SCALE, hover[1] * SCALE, SCALE, SCALE);
    }
  }, [pixels, hover]);

  const paintAt = (clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor((clientX - rect.left) / SCALE);
    const y = Math.floor((clientY - rect.top) / SCALE);
    if (x < 0 || x >= SIZE || y < 0 || y >= SIZE) return;
    const next = [...pixels];
    applyBrush(next, x, y, brushSize, erase);
    onChange(next);
  };

  return (
    <canvas
      ref={canvasRef}
      width={SIZE * SCALE}
      height={SIZE * SCALE}
      className="canvas-grid"
      onContextMenu={(e) => e.preventDefault()}
      onPointerDown={(e) => {
        setPainting(true);
        setErase(e.button === 2);
        paintAt(e.clientX, e.clientY);
      }}
      onPointerMove={(e) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        setHover([
          Math.floor((e.clientX - rect.left) / SCALE),
          Math.floor((e.clientY - rect.top) / SCALE),
        ]);
        if (painting) paintAt(e.clientX, e.clientY);
      }}
      onPointerUp={() => setPainting(false)}
      onPointerLeave={() => {
        setPainting(false);
        setHover(null);
      }}
      onTouchMove={(e) => {
        const touch = e.touches[0];
        if (touch) paintAt(touch.clientX, touch.clientY);
      }}
    />
  );
};

export default CanvasGrid;
