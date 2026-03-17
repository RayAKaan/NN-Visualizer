export function l2Norm(values: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i] * values[i];
  return Math.sqrt(sum);
}

export function mean(values: Float32Array): number {
  if (values.length === 0) return 0;
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += values[i];
  return sum / values.length;
}

export function std(values: Float32Array): number {
  if (values.length === 0) return 0;
  const m = mean(values);
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) {
    const d = values[i] - m;
    sum += d * d;
  }
  return Math.sqrt(sum / values.length);
}

export function sparsity(values: Float32Array, epsilon = 0.01): number {
  if (values.length === 0) return 0;
  let nearZero = 0;
  for (let i = 0; i < values.length; i += 1) {
    if (Math.abs(values[i]) < epsilon) nearZero += 1;
  }
  return nearZero / values.length;
}
