/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        popins: ["Poppins", "sans-serif"],
      },
      colors: {
        primary: "#436850",
        "first-gray": "#64748B",
        "second-gray": "#5F5F5F",
        dark: "#000000",
      },
    },
  },
  plugins: [],
};
