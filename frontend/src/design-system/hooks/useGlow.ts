import { glowStyle } from '@/design-system/tokens/colors';

export function useGlow(color: string, intensity = 0.5) {
  return glowStyle(color, intensity);
}
