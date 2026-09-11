import { useExecutionStore, selectCurrentStage } from "../../../store/executionStore";
import { fmt } from "./visual";
import { MathRenderer } from "../MathRenderer";

export function LiveFormula() {
  const stage = useExecutionStore(selectCurrentStage);
  const trace = useExecutionStore((s) => s.trace);
  const selection = useExecutionStore((s) => s.selection);
  const neuronDetail = useExecutionStore((s) => s.neuronDetail);
  const convDetail = useExecutionStore((s) => s.convDetail);
  const lstmDetail = useExecutionStore((s) => s.lstmDetail);

  if (!stage || !trace) return null;
  const op = stage.operation;
  const act = stage.activation;

  const body = formulaFor(op, act, stage, selection, trace, neuronDetail, convDetail, lstmDetail) ?? undefined;
  if (!body) return null;
  return (
    <div className="rounded-xl border border-[color-mix(in_srgb,var(--accent-secondary)_22%,white)] bg-[color-mix(in_srgb,var(--accent-secondary)_5%,white)] p-3">
      <div className="mb-1 flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-widest text-[var(--accent-secondary)]">
          live math
        </span>
        <span className="text-[10px] text-ink-faint">edited values from this step — computed, not sketched</span>
      </div>
      <div className="text-sm leading-relaxed">
        <MathRenderer latex={body} />
      </div>
    </div>
  );
}

function formulaFor(
  op: string,
  act: string | null | undefined,
  stage: any,
  selection: any,
  trace: any,
  neuronDetail: any,
  convDetail: any,
  lstmDetail: any,
): string | undefined {
  if (neuronDetail && neuronDetail.layerId === stage.layerId && op === "dense") {
    const { preActivation, postActivation, bias, topContributions, activation } = neuronDetail;
    const first = topContributions[0];
    const actName = activation ?? "id";
    const actTex =
      actName === "relu"
        ? `\\mathrm{relu}(z) = \\max(0,z)`
        : actName === "sigmoid"
          ? `\\sigma(z) = (1+e^{-z})^{-1}`
          : actName === "tanh"
            ? `a = \\tanh(z)`
            : `a = z`;
    const contribTex = first
      ? `x_{${first.index}} w_{${first.index},${neuronDetail.neuronIndex}} = ${fmt(first.inputValue, 3)} \\cdot ${fmt(first.weight, 3)} = ${fmt(first.contribution, 4)}`
      : "";
    return `z_{${neuronDetail.neuronIndex}} = \\sum_{i} x_i w_{i,${neuronDetail.neuronIndex}} + b_{${neuronDetail.neuronIndex}} = ${fmt(preActivation, 4)} \\quad (${contribTex}) \\quad ${actTex} \\Longrightarrow a = ${fmt(postActivation, 4)}`;
  }

  if (convDetail && convDetail.layerId === stage.layerId && op === "conv2d") {
    const { preActivation, postActivation, bias, activation } = convDetail;
    const actName = activation ?? "id";
    return `z = \\sum_{p,q,c} x[p,q,c]\\, k_{${convDetail.filterIndex}}[p,q] + b = ${fmt(preActivation, 4)}, \\quad ${actName}(z) = ${fmt(postActivation, 4)}`;
  }

  if (op === "lstm") {
    if (!lstmDetail) return undefined;
    const g = lstmDetail.summary;
    const t = selection.timestep;
    const i = fmt(g.inputMean[t] ?? 0, 3);
    const f = fmt(g.forgetMean[t] ?? 0, 3);
    const o = fmt(g.outputMean[t] ?? 0, 3);
    return `\\text{@ }t=${t}:\\ \\ i_t = \\sigma(x_t W_i + h_{t-1} R_i + b_i) = ${i},\\quad f_t = ${f},\\quad o_t = ${o}`;
  }

  if (op === "dense") {
    return `z = W x + b, \\qquad a = ${act ?? "x"}(z)`;
  }
  if (op === "conv2d") {
    return `z = (x \\ast k) + b, \\qquad a = ${act ?? "x"}(z)`;
  }
  if (op === "max_pool") {
    return `y_{i,j} = \\max_{p,q\\in 2\\times2} x_{2i+p,\\,2j+q}`;
  }
  if (op === "avg_pool") {
    return `y_{i,j} = \\tfrac{1}{4}\\sum_{p,q} x_{2i+p,\\,2j+q}`;
  }
  if (op === "flatten") {
    return `y = \\mathrm{vec}(x)`;
  }
  if (op === "activation") {
    return `${act ?? "σ"}(x) \\text{ pointwise (elementwise)}`;
  }
  if (op === "softmax") {
    return `\\mathrm{softmax}(z)_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}`;
  }
  if (op === "dropout") {
    return `y = x \\quad (\\text{inference: dropout is identity})`;
  }
  return undefined;
}