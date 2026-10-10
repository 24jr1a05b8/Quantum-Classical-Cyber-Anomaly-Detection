/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        quantum: '#7c3aed',
        classical: '#2563eb',
      }
    }
  },
  plugins: []
}
