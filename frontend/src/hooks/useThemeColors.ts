export function useThemeColors() {
  const isDark = !document.documentElement.classList.contains("light");
  const colorMode = document.documentElement.getAttribute("data-color-mode") ?? "default";
  return { isDark, colorMode };
}
