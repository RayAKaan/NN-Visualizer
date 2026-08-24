/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Surfaces - warm barley whites (60% of the UI)
        barley: {
          page: "#FAF7F2",
          wash: "#F6F1E9",
          sunken: "#F3EEE5",
          line: "#E7E0D4",
          linestrong: "#D8CFC0",
        },
        // Text - warm ink scale
        ink: {
          DEFAULT: "#1C1917",
          soft: "#44403C",
          mute: "#57534E",
          faint: "#79716B",
        },
        // Primary accent - dark orange ramp (WCAG: 700=4.84:1, 800=6.83:1 on barley.page)
        ember: {
          50: "#FFF7ED",
          100: "#FFEDD5",
          200: "#FED7AA",
          300: "#FDBA74",
          400: "#FB923C",
          500: "#F97316",
          600: "#EA580C",
          700: "#C2410C",
          800: "#9A3412",
          900: "#7C2D12",
          950: "#431407",
        },
        // Architecture identities - Okabe-Ito derived, colorblind-safe
        arch: {
          ann: "#0072B2",
          cnn: "#00806A",
          rnn: "#A64D85",
        },
        // Semantic status
        status: {
          success: "#15803D",
          successhover: "#166534",
          successbright: "#4ADE80",
          warning: "#B45309",
          danger: "#B91C1C",
          dangerhover: "#991B1B",
          info: "#0072B2",
        },
        // Chart series - Okabe-Ito categorical set
        chart: {
          blue: "#0072B2",
          orange: "#E69F00",
          green: "#009E73",
          purple: "#CC79A7",
          sky: "#56B4E9",
        },
      },
      fontFamily: {
        ui: [
          "-apple-system",
          "BlinkMacSystemFont",
          "SF Pro Text",
          "Segoe UI Variable Text",
          "Segoe UI",
          "system-ui",
          "Roboto",
          "Helvetica Neue",
          "Arial",
          "sans-serif",
        ],
        mono: ["JetBrains Mono", "Fira Code", "SF Mono", "monospace"],
      },
      boxShadow: {
        card: "0 1px 2px rgba(28,25,23,0.04), 0 4px 16px rgba(28,25,23,0.06)",
        pop: "0 2px 4px rgba(28,25,23,0.06), 0 12px 32px rgba(28,25,23,0.12)",
        ember: "0 1px 2px rgba(194,65,12,0.25), 0 4px 14px rgba(194,65,12,0.18)",
      },
    },
  },
  plugins: [],
}
