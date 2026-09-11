import { selectCurrentStage, useExecutionStore } from "../../../store/executionStore";
import { fmt, fmtMs, StatChips, opLabel } from "./visual";

export function ExecutionInspector() {
  const stage = useExecutionStore(selectCurrentStage);
  const trace = useExecutionStore((s) => s.trace);
  const neuronDetail = useExecutionStore((s) => s.neuronDetail);
  const convDetail = useExecutionStore((s) => s.convDetail);
  const selection = useExecutionStore((s) => s.selection);

  const stageActive = Boolean(trace && stage);

  return (
    <aside className="flex h-full min-h-0 flex-col">
      <div className="flex items-center justify-between px-1 pb-2">
        <div className="text-[10px] font-semibold uppercase tracking-widest text-ink-faint">inspector</div>
        {stageActive && stage?.compute_time_ms != null ? (
          <span className="font-mono text-[11px] text-ink-faint">{fmtMs(stage.compute_time_ms)}</span>
        ) : null}
      </div>

      {!trace ? (
        <div className="rounded-xl border border-dashed border-barley-line p-4 text-center text-sm text-ink-faint">
          Run the trace to inspect real tensors.
          <div className="mt-1 text-[11px]">values here come from the live backend — nothing is fabricated.</div>
        </div>
      ) : !stage ? (
        <div className="rounded-xl border border-dashed border-barley-line p-4 text-center text-sm text-ink-faint">
          Pick a stage in the graph or timeline.
        </div>
      ) : (
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
          <div className="rounded-xl border border-barley-line bg-barley-page/70 p-2.5">
            <div className="mb-1 flex items-center justify-between">
              <span className="truncate text-[13px] font-semibold text-ink">{stage.name}</span>
              <span className="shrink-0 text-[11px] text-ink-faint">{opLabel(stage.operation)}</span>
            </div>
            <div className="text-[11px] text-ink-mute">
              {stage.activation ? `activation ${stage.activation} · ` : ""}
              in {stage.inputShape.join("×")} → out {stage.outputShape.join("×")}
              {stage.params != null ? ` · ${stage.params.toLocaleString()} params` : ""}
            </div>
          </div>

          {stage.operation === "input" && stage.inputShape.length === 2 && trace ? (
            <ValueGrid
              title={`Sample tensor — shape ${trace.input.shape.join("×")}`}
              data={trace.input.values}
              shape={trace.input.shape}
            />
          ) : stage.operation === "conv2d" && stage.feature_maps ? (
            <div className="space-y-3">
              <ValueGrid
                title={`Channel ${selection.filterIndex} — ${stage.outputShape.join("×")}`}
                data={stage.feature_maps[selection.filterIndex] ?? []}
                shape={[stage.outputShape[0], stage.outputShape[1]]}
              />
              {neuronDetail && stage.operation === "conv2d" ? (
                <div className="rounded-xl border border-barley-line bg-barley-page/70 p-2.5">
                  <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-ink-faint">selection</div>
                  <div className="text-[11px] text-ink-mute">hover/click a cell to slide the trace.</div>
                </div>
              ) : null}
            </div>
          ) : stage.operation === "lstm" && neuronDetail === null ? (
            <div className="space-y-3">
              <ValueGrid
                title={`t = ${selection.timestep} · cell norm`}
                data={stage.output_data ?? []}
                shape={[stage.outputShape[0] ?? 1]}
              />
              <CellNormChip timestep={selection.timestep} />
              <blockquote className="border-l-2 border-[var(--accent-quaternary)] pl-2 text-[11px] text-ink-mute">
                gate curves, per-unit sliders, and verified cell math are in <b>ExecutionGraph → stage view</b>.
              </blockquote>
            </div>
          ) : stage.operation === "lstm" ? (
            <ValueGrid
              title={`t = ${selection.timestep} · output units`}
              data={stage.output_data ?? []}
              shape={[stage.outputShape[0] ?? 1]}
            />
          ) : (
            <ValueGrid
              title={stage.operation === "flatten" ? "Flattened values" : "Output tensor"}
              data={stage.output_data ?? []}
              shape={stage.outputShape}
            />
          )}

          {stage.statistics ? <StatChips stats={stage.statistics} /> : null}
        </div>
      )}
    </aside>
  );
}

function ValueGrid({
  title,
  data,
  shape,
  onCell,
}: {
  title: string;
  data: number[] | number[][];
  shape: number[];
  onCell?: (h: number, w: number) => void;
}) {
  const twoD = Array.isArray(data[0]);
  const rows = twoD ? (data as number[][]) : makeRows(data as number[], shape);
  const rowCap = 24;
  const colCap = 20;
  const grid = rows.slice(0, rowCap).map((r) => r.slice(0, colCap));
  const clipped = rows.length > rowCap || rows[0].length > colCap;

  return (
    <div className="rounded-xl border border-barley-line bg-barley-page/70 p-2.5">
      <div className="mb-1.5 flex items-center justify-between text-[11px]">
        <span className="font-semibold uppercase tracking-wide text-ink-faint">{title}</span>
        <span className="font-mono text-ink-faint">
          {shape.join("×")}
        </span>
      </div>
      {clipped ? (
        <div className="mb-1 text-[10px] text-ink-faint">
          showing first {grid.length}×{grid[0]?.length ?? 0} cells — hover for values
        </div>
      ) : null}
      <div className="grid gap-px" style={{ gridTemplateColumns: `repeat(${grid[0]?.length ?? 1}, 26px)` }}>
        {grid.map((row, r) =>
          row.map((v, c) => (
            <div
              key={`${r}-${c}`}
              title={`[${r},${c}] = ${fmt(v)}`}
              onPointerEnter={onCell ? () => onCell(r, c) : undefined}
              className="grid h-[26px] w-[26px] place-items-center rounded-[3px] font-mono text-[10px]"
              style={{
                background:
                  v > 0
                    ? `rgba(194,65,12,${0.06 + Math.min(1, Math.abs(v) * 2) * 0.6})`
                    : v < 0
                      ? `rgba(0,114,178,${0.06 + Math.min(1, Math.abs(v) * 2) * 0.6})`
                      : "rgba(236,231,222,0.5)",
                color: Math.abs(v) > 0.35 ? "rgba(255,255,255,0.92)" : "var(--ink)",
              }}
            >
              {fmt(v, 1)}
            </div>
          )),
        )}
      </div>
    </div>
  );
}

function makeRows(data: number[], shape: number[]): number[][] {
  const cols = (shape[shape.length - 1] ?? 1) || 1;
  const rows: number[][] = [];
  const rowsLen = shape.length >= 2 ? (shape[shape.length - 2] ?? 1) : 1;
  const total = (rowsLen || 1) * cols;
  for (let r = 0; r < rowsLen; r++) {
    const start = r * cols;
    const end = Math.min(start + cols, data.length);
    if (start >= data.length) break;
    rows.push(data.slice(start, end));
  }
  return rows.length ? rows : [data];
}

function CellNormChip({ timestep }: { timestep: number }) {
  const lstmDetail = useExecutionStore((s) => s.lstmDetail);
  const cellNorm = lstmDetail?.summary.cellNorm ?? [];
  if (!cellNorm.length) return null;
  return (
    <div className="rounded-xl border border-barley-line bg-barley-page/70 p-2.5 text-[11px] text-ink-mute">
      cell ‖c‖ at t = <b className="font-mono text-ink">{fmt(cellNorm[timestep] ?? 0)}</b>
    </div>
  );
}