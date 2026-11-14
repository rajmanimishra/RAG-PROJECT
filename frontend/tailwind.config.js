/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#17211f",
        muted: "#687572",
        canvas: "#f4f6f2",
        forest: "#176b55",
      },
      boxShadow: {
        card: "0 18px 60px rgba(28, 51, 43, 0.08)",
      },
    },
  },
  plugins: [],
};
