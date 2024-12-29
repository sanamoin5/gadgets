import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import Container from "@mui/material/Container";
import Icon from "@mui/material/Icon";

// Material Kit components
import MKBox from "../../../components/MKBox";
import MKTypography from "../../../components/MKTypography";
import MKButton from "../../../components/MKButton";

// Navbar components
import DefaultNavbarDropdown from "./DefaultNavbarDropdown";
import DefaultNavbarMobile from "./DefaultNavbarMobile";

// Material Kit base styles
import breakpoints from "../../../assets/theme/base/breakpoints";

function DefaultNavbar({ brand, routes, transparent, light, action }) {
  const [mobileNavbar, setMobileNavbar] = useState(false);
  const [mobileView, setMobileView] = useState(false);

  const toggleMobileNavbar = () => setMobileNavbar((prev) => !prev);

  useEffect(() => {
    // Adjust view based on screen size
    function displayMobileNavbar() {
      if (window.innerWidth < breakpoints.values.lg) {
        setMobileView(true);
        setMobileNavbar(false);
      } else {
        setMobileView(false);
        setMobileNavbar(false);
      }
    }

    // Add resize listener
    window.addEventListener("resize", displayMobileNavbar);

    // Initial setup
    displayMobileNavbar();

    // Cleanup listener
    return () => window.removeEventListener("resize", displayMobileNavbar);
  }, []);

  const renderNavbarItems = routes.map(({ name, route }) => (
    <MKBox
      key={name}
      component={Link}
      to={route}
      onClick={() => setMobileNavbar(false)} // Close mobile menu on navigation
      sx={{
        textDecoration: "none",
        py: 1,
        px: 2,
        "&:hover": {
          backgroundColor: "rgba(0, 0, 0, 0.05)",
        },
      }}
    >
      <MKTypography variant="button" fontWeight="regular" textTransform="capitalize" color="dark">
        {name}
      </MKTypography>
    </MKBox>
  ));

  return (
    <MKBox
      py={1}
      px={{ xs: 2, sm: 3, lg: 4 }}
      width="100%"
      position="fixed"
      top={0}
      left={0}
      zIndex={1000}
      bgcolor={transparent ? "transparent" : "white"}
      sx={{
        boxShadow: transparent ? "none" : "0px 2px 4px rgba(0, 0, 0, 0.1)",
        backdropFilter: transparent ? "none" : "blur(10px)",
      }}
    >
      <Container>
        <MKBox display="flex" justifyContent="space-between" alignItems="center">
          {/* Brand/Logo */}
          <MKBox component={Link} to="/" lineHeight={1}>
            <MKTypography variant="button" fontWeight="bold" color={light ? "white" : "dark"}>
              {brand}
            </MKTypography>
          </MKBox>

          {/* Desktop Navbar */}
          <MKBox display={{ xs: "none", lg: "flex" }} ml="auto">
            {renderNavbarItems}
          </MKBox>

          {/* Action Button */}
          {action && (
            <MKBox ml={2}>
              {action.type === "internal" ? (
                <MKButton
                  component={Link}
                  to={action.route}
                  variant="gradient"
                  color={action.color}
                  size="small"
                >
                  {action.label}
                </MKButton>
              ) : (
                <MKButton
                  component="a"
                  href={action.route}
                  target="_blank"
                  rel="noreferrer"
                  variant="gradient"
                  color={action.color}
                  size="small"
                >
                  {action.label}
                </MKButton>
              )}
            </MKBox>
          )}

          {/* Mobile Navbar Icon */}
          <MKBox
            display={{ xs: "inline-block", lg: "none" }}
            onClick={toggleMobileNavbar}
            sx={{ cursor: "pointer" }}
          >
            <Icon>{mobileNavbar ? "close" : "menu"}</Icon>
          </MKBox>
        </MKBox>

        {/* Mobile Navbar */}
        {mobileView && (
          <DefaultNavbarMobile
            routes={routes}
            open={mobileNavbar}
            onClose={() => setMobileNavbar(false)} // Pass close handler
          />
        )}
      </Container>
    </MKBox>
  );
}

// Setting default props for DefaultNavbar
DefaultNavbar.defaultProps = {
  brand: "AIra",
  transparent: false,
  light: false,
  action: null,
};

// PropTypes for validation
DefaultNavbar.propTypes = {
  brand: PropTypes.string,
  routes: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      route: PropTypes.string.isRequired,
    })
  ).isRequired,
  transparent: PropTypes.bool,
  light: PropTypes.bool,
  action: PropTypes.shape({
    type: PropTypes.oneOf(["external", "internal"]),
    route: PropTypes.string.isRequired,
    color: PropTypes.string.isRequired,
    label: PropTypes.string.isRequired,
  }),
};

export default DefaultNavbar;
