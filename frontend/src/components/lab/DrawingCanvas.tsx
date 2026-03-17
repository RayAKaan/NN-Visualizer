import { useCallback, useEffect, useRef } from "react";
import { useLabStore } from "../../store/labStore";

const DISPLAY = 280;
const INTERNAL = 560;

export function DrawingCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const last = useRef<{ x: number; y: number } | null>(null);
  const setInputPixels = useLabStore((s) => s.setInputPixels);

  const exportPixels = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const small = document.createElement("canvas");
    small.width = 28;
    small.height = 28;
    const sctx = small.getContext("2d");
    if (!sctx) return;
    sctx.fillStyle = "black";
    sctx.fillRect(0, 0, 28, 28);
    sctx.imageSmoothingEnabled = true;
    sctx.drawImage(cv, 0, 0, 28, 28);
    const data = sctx.getImageData(0, 0, 28, 28).data;
    const out = new Float32Array(28 * 28);
    for (let i = 0; i < out.length; i += 1) out[i] = data[i * 4] / 255;
    setInputPixels(out);
  }, [setInputPixels]);

  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, INTERNAL, INTERNAL);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 34;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    exportPixels();
  }, [exportPixels]);

  const clear = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, INTERNAL, INTERNAL);
    exportPixels();
  };

  const drawAt = (clientX: number, clientY: number) => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext("2d");
    if (!cv || !ctx || !drawing.current) return;
    const rect = cv.getBoundingClientRect();
    const scale = cv.width / rect.width;
    const x = (clientX - rect.left) * scale;
    const y = (clientY - rect.top) * scale;
    if (!last.current) {
      last.current = { x, y };
      return;
    }
    const midX = (last.current.x + x) / 2;
    const midY = (last.current.y + y) / 2;
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.quadraticCurveTo(last.current.x, last.current.y, midX, midY);
    ctx.stroke();
    last.current = { x, y };
    exportPixels();
  };

  return (
    <div className="space-y-2">
      <div className="relative h-[280px] w-[280px] rounded-2xl border border-cyan-400/35 bg-black shadow-[inset_0_0_34px_rgba(34,211,238,0.16)]">
        <canvas
          ref={canvasRef}
          width={INTERNAL}
          height={INTERNAL}
          className="absolute inset-0 h-full w-full cursor-crosshair rounded-2xl"
          onMouseDown={(e) => {
            drawing.current = true;
            drawAt(e.clientX, e.clientY);
          }}
          onMouseMove={(e) => drawAt(e.clientX, e.clientY)}
          onMouseUp={() => {
            drawing.current = false;
            last.current = null;
            exportPixels();
          }}
          onMouseLeave={() => {
            drawing.current = false;
            last.current = null;
          }}
          aria-label="Drawing canvas for lab"
          role="img"
        />
        <div
          className="pointer-events-none absolute inset-0 rounded-2xl"
          style={{
            backgroundImage:
              "linear-gradient(to right, rgba(255,255,255,0.04) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.04) 1px, transparent 1px)",
            backgroundSize: "calc(100% / 28) calc(100% / 28)",
          }}
        />
      </div>
      <button
        type="button"
        onClick={clear}
        className="h-8 rounded-lg border border-rose-400/40 bg-rose-500/10 px-3 text-xs text-rose-200"
      >
        Clear
      </button>
    </div>
  );
}
