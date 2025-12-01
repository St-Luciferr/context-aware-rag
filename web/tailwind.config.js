/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f4ff',
          100: '#e0e9ff',
          200: '#c7d6fe',
          300: '#a3b8fc',
          400: '#7a8ff8',
          500: '#5a67f2',
          600: '#4549e6',
          700: '#3a3bcb',
          800: '#3133a4',
          900: '#2e3182',
        },
        dark: {
          50: '#f6f6f7',
          100: '#e2e3e5',
          200: '#c5c6ca',
          300: '#a0a2a8',
          400: '#7c7e86',
          500: '#61636b',
          600: '#4d4e55',
          700: '#3f4046',
          800: '#27272a',
          900: '#18181b',
          950: '#09090b',
        }
      },
    },
  },
  plugins: [],
};