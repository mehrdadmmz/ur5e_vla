/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        'block-table': '#4ade80',
        'block-stacked': '#60a5fa',
        'block-held': '#fbbf24',
        'block-moving': '#f87171',
      },
    },
  },
  plugins: [],
}
