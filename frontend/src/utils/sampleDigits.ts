const blank = () => Array.from({ length: 784 }, () => 0);

const setPixel = (grid: number[], x: number, y: number, value = 1) => {
  if (x < 0 || x > 27 || y < 0 || y > 27) return;
  grid[y * 28 + x] = value;
};

const drawLine = (grid: number[], x1: number, y1: number, x2: number, y2: number, value = 1) => {
  const steps = Math.max(Math.abs(x2 - x1), Math.abs(y2 - y1));
  for (let i = 0; i <= steps; i += 1) {
    const x = Math.round(x1 + ((x2 - x1) * i) / steps);
    const y = Math.round(y1 + ((y2 - y1) * i) / steps);
    setPixel(grid, x, y, value);
  }
};

const makeDigit = (digit: number) => {
  const grid = blank();
  switch (digit) {
    case 0:
      drawLine(grid, 8, 6, 20, 6); drawLine(grid, 8, 22, 20, 22); drawLine(grid, 8, 6, 8, 22); drawLine(grid, 20, 6, 20, 22); break;
    case 1:
      drawLine(grid, 14, 6, 14, 22); drawLine(grid, 11, 9, 14, 6); break;
    case 2:
      drawLine(grid, 8, 7, 20, 7); drawLine(grid, 20, 7, 20, 14); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 8, 14, 8, 22); drawLine(grid, 8, 22, 20, 22); break;
    case 3:
      drawLine(grid, 8, 7, 20, 7); drawLine(grid, 20, 7, 20, 22); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 8, 22, 20, 22); break;
    case 4:
      drawLine(grid, 8, 6, 8, 14); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 20, 6, 20, 22); break;
    case 5:
      drawLine(grid, 8, 6, 20, 6); drawLine(grid, 8, 6, 8, 14); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 20, 14, 20, 22); drawLine(grid, 8, 22, 20, 22); break;
    case 6:
      drawLine(grid, 8, 6, 8, 22); drawLine(grid, 8, 6, 20, 6); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 20, 14, 20, 22); drawLine(grid, 8, 22, 20, 22); break;
    case 7:
      drawLine(grid, 8, 6, 20, 6); drawLine(grid, 20, 6, 12, 22); break;
    case 8:
      drawLine(grid, 8, 6, 20, 6); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 8, 22, 20, 22); drawLine(grid, 8, 6, 8, 22); drawLine(grid, 20, 6, 20, 22); break;
    case 9:
      drawLine(grid, 8, 6, 20, 6); drawLine(grid, 8, 6, 8, 14); drawLine(grid, 8, 14, 20, 14); drawLine(grid, 20, 6, 20, 22); break;
  }
  return grid;
};

export const sampleDigits: Record<string, number[]> = Object.fromEntries(
  Array.from({ length: 10 }, (_, digit) => [String(digit), makeDigit(digit)])
);
