import { createTheme } from "@mui/material/styles";

const darkTheme = createTheme({
  palette: {
    mode: "dark",
    primary: {
      main: "#006d77", // Dark teal
    },
    secondary: {
      main: "#ff477e", // Magenta
    },
    background: {
      default: "#121212", // Midnight blue
      paper: "#1c1c1e", // Charcoal black
    },
    text: {
      primary: "#e5e5e5", // Light grey
      secondary: "#9d4edd", // Bright neon purple for accents
    },
  },
  typography: {
    fontFamily: "'Poppins', 'Roboto', sans-serif", // Trendy font
    h1: { fontWeight: 700 },
    h2: { fontWeight: 600 },
    body1: { fontWeight: 400 },
    button: { textTransform: "none", fontWeight: 500 },
  },
  shape: {
    borderRadius: 12, // Subtle rounded corners
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 8,
          padding: "8px 16px",
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundColor: "#1c1c1e", // Matches paper background
        },
      },
    },
  },
});

export default darkTheme;
