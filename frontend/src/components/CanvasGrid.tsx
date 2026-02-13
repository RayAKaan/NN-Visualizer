import React, { useEffect, useRef, useState } from "react";

const GRID_SIZE = 28;
const SCALE = 10;

interface CanvasGridProps {
  pixels: number[];
  onChange: (next: number[]) => void;
}

const CanvasGrid: React.FC<CanvasGridProps> = ({ pixels, onChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isPainting, setIsPainting] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#f2f2f4";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let row = 0; row < GRID_SIZE; row += 1) {
      for (let col = 0; col < GRID_SIZE; col += 1) {
        const idx = row * GRID_SIZE + col;
        if (pixels[idx] > 0) {
          ctx.fillStyle = "#1f2430";
          ctx.fillRect(col * SCALE, row * SCALE, SCALE, SCALE);
        }
      }
    }

    ctx.strokeStyle = "rgba(31,36,48,0.08)";
    for (let i = 0; i <= GRID_SIZE; i += 1) {
      ctx.beginPath();
      ctx.moveTo(i * SCALE, 0);
      ctx.lineTo(i * SCALE, GRID_SIZE * SCALE);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, i * SCALE);
      ctx.lineTo(GRID_SIZE * SCALE, i * SCALE);
      ctx.stroke();
    }
  }, [pixels]);

  const updatePixel = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const col = Math.floor(x / SCALE);
    const row = Math.floor(y / SCALE);
    if (row < 0 || row >= GRID_SIZE || col < 0 || col >= GRID_SIZE) return;
    const next = [...pixels];
    next[row * GRID_SIZE + col] = 1;
    onChange(next);
  };

  return (
    <canvas
      ref={canvasRef}
      className="canvas-grid"
      width={GRID_SIZE * SCALE}
      height={GRID_SIZE * SCALE}
      onPointerDown={(event) => {
        setIsPainting(true);
        updatePixel(event);
      }}
      onPointerMove={(event) => {
        if (isPainting) {
          updatePixel(event);
        }
      }}
      onPointerUp={() => setIsPainting(false)}
      onPointerLeave={() => setIsPainting(false)}
    />
  );
};

export default CanvasGrid;
