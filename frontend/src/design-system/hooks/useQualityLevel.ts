import { useEffect, useState } from 'react';

export type QualityLevel = 'low' | 'medium' | 'high' | 'ultra';

const STORAGE_KEY = 'nnlab-quality-level';

export function useQualityLevel(defaultLevel: QualityLevel = 'high') {
  const [level, setLevel] = useState<QualityLevel>(() => {
    if (typeof window === 'undefined') return defaultLevel;
    return (localStorage.getItem(STORAGE_KEY) as QualityLevel) || defaultLevel;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    localStorage.setItem(STORAGE_KEY, level);
  }, [level]);

  return { level, setLevel } as const;
}
