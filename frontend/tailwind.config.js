/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/app/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#E33CF8", // magenta-ish
          50: "#FBECFF",
          100: "#F8D9FB",
          500: "#E33CF8",
          700: "#C12EE0",
        },
        accent: {
          DEFAULT: "#FF4DA6",
        },
        bg: {
          DEFAULT: "#07060a", // almost black
          700: "#0b0b0d",
        },
        glass: {
          light: "rgba(255,255,255,0.06)",
          lighter: "rgba(255,255,255,0.03)",
          dark: "rgba(0,0,0,0.45)",
        },
      },
      boxShadow: {
        "glass-lg": "0 8px 30px rgba(0,0,0,0.6)",
        "neon": "0 10px 40px rgba(227,60,248,0.18)",
      },
      backdropBlur: {
        xs: "2px",
        sm: "4px",
        md: "8px",
      },
      spacing: {
        "128": "32rem",
      },
      fontFamily: {
        sans: ["Inter", "ui-sans-serif", "system-ui", "Helvetica", "Arial"],
        display: ["Poppins", "Inter", "sans-serif"],
      },
      borderRadius: {
        "xl-2": "18px",
      },
    },
  },
  plugins: [],
};
