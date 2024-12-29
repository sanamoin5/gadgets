import React from "react";
import { useNavigate } from "react-router-dom";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import SimpleFooter from "examples/Footers/SimpleFooter";

function ErrorPage() {
  const navigate = useNavigate();

  const handleGoHome = () => {
    navigate("/");
  };

  return (
    <>
      <MKBox
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="80vh"
        px={2}
      >
        <MKTypography variant="h3" mb={2}>
          Oops! Something went wrong.
        </MKTypography>
        <MKButton variant="gradient" color="info" onClick={handleGoHome}>
          Go to Home Page
        </MKButton>
      </MKBox>
      <SimpleFooter />
    </>
  );
}

export default ErrorPage;
