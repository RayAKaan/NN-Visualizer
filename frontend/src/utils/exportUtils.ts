export function exportPrediction(pixels: number[], prediction: number, confidence: number, probabilities: number[]) {
  const c = document.createElement("canvas");
  c.width = 640; c.height = 360;
  const ctx = c.getContext("2d")!;
  ctx.fillStyle = "#0a0e17"; ctx.fillRect(0, 0, c.width, c.height);
  ctx.fillStyle = "#fff"; ctx.font = "20px Inter";
  ctx.fillText(`Prediction: ${prediction}`, 24, 36);
  ctx.fillText(`Confidence: ${(confidence * 100).toFixed(1)}%`, 24, 68);
  probabilities.forEach((p, i) => { ctx.fillStyle = i === prediction ? "#06b6d4" : "#64748b"; ctx.fillRect(24, 100 + i * 22, p * 280, 14); ctx.fillStyle = "#fff"; ctx.fillText(String(i), 312, 112 + i * 22); });
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) { const v = Math.round((pixels[i] || 0) * 255); img.data[i * 4 + 0] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255; }
  const temp = document.createElement("canvas"); temp.width = 28; temp.height = 28; temp.getContext("2d")!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false; ctx.drawImage(temp, 380, 90, 224, 224);
  const a = document.createElement("a"); a.href = c.toDataURL("image/png"); a.download = `prediction-${Date.now()}.png`; a.click();
}
