import React, { useEffect, useRef, useCallback } from "react";

const GRID_SIZE = 28;
const SCALE = 10;
const BRUSH_RADIUS = 1.6;
const BRUSH_STRENGTH = 0.65;

interface CanvasGridProps {
  pixels: number[];
  onChange: (next: number[]) => void;
}

const CanvasGrid: React.FC<CanvasGridProps> = ({ pixels, onChange }) => {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const ctxRef = useRef<CanvasRenderingContext2D | null>(null);

  const isPaintingRef = useRef(false);
  const lastPosRef = useRef<{ x: number; y: number } | null>(null);
  const workingPixelsRef = useRef<number[]>(pixels);

  /* --------------------------------------------------
     Sync pixel buffer
  -------------------------------------------------- */
  useEffect(() => {
    workingPixelsRef.current = pixels;
  }, [pixels]);

  /* --------------------------------------------------
     Canvas setup
  -------------------------------------------------- */
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctxRef.current = ctx;
    ctx.imageSmoothingEnabled = false;
  }, []);

  /* --------------------------------------------------
     Render canvas (only when pixels change)
  -------------------------------------------------- */
  useEffect(() => {
    const ctx = ctxRef.current;
    if (!ctx) return;

    ctx.clearRect(0, 0, GRID_SIZE * SCALE, GRID_SIZE * SCALE);

    // background
    ctx.fillStyle = "#f2f2f4";
    ctx.fillRect(0, 0, GRID_SIZE * SCALE, GRID_SIZE * SCALE);

    // pixels
    for (let r = 0; r < GRID_SIZE; r++) {
      for (let c = 0; c < GRID_SIZE; c++) {
        const v = pixels[r * GRID_SIZE + c];
        if (v <= 0) continue;

        ctx.fillStyle = `rgba(31,36,48,${v})`;
        ctx.fillRect(c * SCALE, r * SCALE, SCALE, SCALE);
      }
    }

    // grid
    ctx.strokeStyle = "rgba(31,36,48,0.06)";
    ctx.beginPath();
    for (let i = 0; i <= GRID_SIZE; i++) {
      ctx.moveTo(i * SCALE, 0);
      ctx.lineTo(i * SCALE, GRID_SIZE * SCALE);
      ctx.moveTo(0, i * SCALE);
      ctx.lineTo(GRID_SIZE * SCALE, i * SCALE);
    }
    ctx.stroke();
  }, [pixels]);

  /* --------------------------------------------------
     Brush logic (batched & stable)
  -------------------------------------------------- */
  const applyBrush = useCallback((gx: number, gy: number) => {
    const next = [...workingPixelsRef.current];

    const r0 = Math.floor(gy - BRUSH_RADIUS);
    const r1 = Math.ceil(gy + BRUSH_RADIUS);
    const c0 = Math.floor(gx - BRUSH_RADIUS);
    const c1 = Math.ceil(gx + BRUSH_RADIUS);

    for (let r = r0; r <= r1; r++) {
      for (let c = c0; c <= c1; c++) {
        if (r < 0 || r >= GRID_SIZE || c < 0 || c >= GRID_SIZE) continue;

        const dx = c + 0.5 - gx;
        const dy = r + 0.5 - gy;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist > BRUSH_RADIUS) continue;

        const falloff = 1 - dist / BRUSH_RADIUS;
        const idx = r * GRID_SIZE + c;

        next[idx] = Math.min(
          1,
          next[idx] + falloff * BRUSH_STRENGTH
        );
      }
    }

    workingPixelsRef.current = next;
    onChange(next);
  }, [onChange]);

  const paintAtEvent = useCallback(
    (e: React.PointerEvent<HTMLCanvasElement>) => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const rect = canvas.getBoundingClientRect();
      const x = (e.clientX - rect.left) / SCALE;
      const y = (e.clientY - rect.top) / SCALE;

      if (lastPosRef.current) {
        const { x: lx, y: ly } = lastPosRef.current;
        const steps = Math.max(
          Math.abs(x - lx),
          Math.abs(y - ly),
          1
        );

        for (let i = 0; i <= steps; i++) {
          const t = i / steps;
          applyBrush(
            lx + (x - lx) * t,
            ly + (y - ly) * t
          );
        }
      } else {
        applyBrush(x, y);
      }

      lastPosRef.current = { x, y };
    },
    [applyBrush]
  );

  /* --------------------------------------------------
     Pointer handlers
  -------------------------------------------------- */
  return (
    <canvas
      ref={canvasRef}
      className="canvas-grid"
      width={GRID_SIZE * SCALE}
      height={GRID_SIZE * SCALE}
      onPointerDown={(e) => {
        isPaintingRef.current = true;
        lastPosRef.current = null;
        paintAtEvent(e);
      }}
      onPointerMove={(e) => {
        if (isPaintingRef.current) paintAtEvent(e);
      }}
      onPointerUp={() => {
        isPaintingRef.current = false;
        lastPosRef.current = null;
      }}
      onPointerLeave={() => {
        isPaintingRef.current = false;
        lastPosRef.current = null;
      }}
    />
  );
};

export default React.memo(CanvasGrid);