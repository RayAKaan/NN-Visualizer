export const ambientVertex = `
attribute vec2 position;
varying vec2 vUv;
void main() {
  vUv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}
`;

export const ambientFragment = `
precision mediump float;

varying vec2 vUv;

uniform float uTime;
uniform vec2 uResolution;
uniform float uState;
uniform vec2 uMouse;
uniform float uBrightness;

float hash(vec2 p) {
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p) {
  vec2 i = floor(p);
  vec2 f = fract(p);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

float grid(vec2 uv, float scale) {
  vec2 g = fract(uv * scale);
  float line = step(g.x, 0.01) + step(g.y, 0.01);
  return line;
}

void main() {
  vec2 uv = vUv;
  float t = uTime * 0.05;
  float n = noise(uv * 3.5 + t) * 0.6 + noise(uv * 12.0 - t) * 0.4;
  float g = grid(uv + vec2(t * 0.02, t * 0.02), 28.0);
  float ripple = exp(-distance(uv, uMouse) * 12.0) * 0.35;

  float pulse = 0.0;
  if (uState > 1.5 && uState < 2.5) {
    pulse = sin(uTime * 2.0) * 0.5 + 0.5;
  }
  float intensity = (n * 0.035 + g * 0.02 + ripple * 0.02 + pulse * 0.02) * uBrightness;

  vec3 base = vec3(0.05, 0.06, 0.08);
  vec3 accent = vec3(0.08, 0.15, 0.25);
  vec3 color = mix(base, accent, intensity * 6.0);
  gl_FragColor = vec4(color * intensity, 1.0);
}
`;
