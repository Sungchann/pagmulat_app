module.exports = {
  content: [
    './src/**/*.{html,ts}',
  ],
  theme: {
    extend: {
      colors: {
        'primary-blue': '#3b82f6',
        'secondary-gray': '#e2e8f0',
        'base': '#f3f4f6', // Slightly darker than gray-50
        'dark-text': '#1a202c',
        'light-gray-border': '#dbe2ed',
        'table-header-bg': '#f8fafc',
        'table-hover-bg': '#f0f4f8',
        'table-text': '#4a5568',
        'hover-blue-dark': '#3366cc',
      },
      boxShadow: {
        'light': '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
        'hover': '0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
      },
      fontFamily: {
        sans: ['Poppins', 'sans-serif'],
      }
    }
  },
  plugins: [],
};
