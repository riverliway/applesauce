import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./index.html",
    "./src/index.css",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
};
export default config;
