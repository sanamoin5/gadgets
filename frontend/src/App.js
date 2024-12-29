// src/App.js

import React from "react";
import { ThemeProvider } from "@mui/material/styles";
import CssBaseline from "@mui/material/CssBaseline";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

// Import your theme
import theme from "./assets/theme";

// Import routes and navigation links as named exports
import { routes, navLinks } from "./routes";

// Import Navbar and Footer components
import DefaultNavbar from "./examples/Navbars/DefaultNavbar";
import SimpleFooter from "./examples/Footers/SimpleFooter";

// Import the Error Page
import ErrorPage from "./pages/ErrorPage"; // Ensure this path is correct

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Router>
        {/* Navbar */}
        <DefaultNavbar
          brand="AIra" // Replace with your brand name or logo
          routes={navLinks} // Navigation links: Home, Take the Quiz, All Gadgets
          transparent={false}
          light={false}
        />

        {/* Application Routes */}
        <Routes>
          {routes.map((route) => (
            <Route key={route.key} path={route.route} element={route.component} />
          ))}
          {/* Wildcard route for handling undefined paths */}
          <Route path="*" element={<ErrorPage />} />
        </Routes>

        {/* Footer */}
        <SimpleFooter
          company={{ href: "https://sanamoin5.github.io/", name: "Sana" }}
          links={[
            { href: "https://sanamoin5.github.io/", name: "About Me" },
            { href: "/contact", name: "Contact" },
            { href: "https://github.com/sanamoin5/gadgets", name: "GitHub" },
            { href: "/license", name: "License" },
          ]}
          light={false}
        />
      </Router>
    </ThemeProvider>
  );
}

export default App;

