export const syntheticDigits: Record<number, number[]> = {
  0: Array(784).fill(0).map((_, i) => ((Math.floor(i / 28) === 4 || Math.floor(i / 28) === 23 || i % 28 === 4 || i % 28 === 23) ? 1 : 0)),
  1: Array(784).fill(0).map((_, i) => (i % 28 > 12 && i % 28 < 16 ? 1 : 0)),
};

export async function loadSampleDigits(): Promise<Record<number, number[]>> {
  try {
    const data = await fetch("http://localhost:8000/samples").then((r) => r.json());
    return Object.fromEntries(Object.entries(data).map(([k, v]) => [Number(k), v as number[]]));
  } catch {
    return syntheticDigits;
  }
}
