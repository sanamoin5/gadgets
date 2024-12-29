import { Link } from "react-router-dom";
import PropTypes from "prop-types";
import Collapse from "@mui/material/Collapse";
import MKBox from "../../../components/MKBox";
import MKTypography from "../../../components/MKTypography";

function DefaultNavbarMobile({ routes, open, onClose }) {
  return (
    <Collapse in={Boolean(open)} timeout="auto" unmountOnExit>
      <MKBox
        width="100%"
        sx={{
          bgcolor: "rgba(255, 255, 255, 0.95)", // Semi-transparent white background
          position: "absolute",
          top: "100%",
          left: 0,
          zIndex: 9999,
          boxShadow: "0px 4px 6px rgba(0, 0, 0, 0.1)", // Subtle shadow for the dropdown
          borderRadius: "0 0 8px 8px", // Rounded corners at the bottom
          backdropFilter: "blur(10px)", // Blur effect
          py: 2,
          px: 1,
          transition: "all 0.3s ease", // Smooth transition
        }}
      >
        {routes.map(({ name, route }) => (
          <MKBox
            key={name}
            component={Link}
            to={route}
            onClick={onClose} // Close dropdown when clicked
            sx={{
              textDecoration: "none",
              width: "100%",
              display: "block",
              py: 1.5,
              px: 2,
              "&:hover": {
                backgroundColor: "rgba(0, 0, 0, 0.05)",
              },
            }}
          >
            <MKTypography
              variant="button"
              fontWeight="regular"
              textTransform="capitalize"
              color="dark"
            >
              {name}
            </MKTypography>
          </MKBox>
        ))}
      </MKBox>
    </Collapse>
  );
}

// PropTypes for validation
DefaultNavbarMobile.propTypes = {
  routes: PropTypes.arrayOf(
    PropTypes.shape({
      name: PropTypes.string.isRequired,
      route: PropTypes.string.isRequired,
    })
  ).isRequired,
  open: PropTypes.bool.isRequired,
  onClose: PropTypes.func.isRequired, // Trigger for closing the menu
};

export default DefaultNavbarMobile;
