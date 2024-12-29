import React from "react";
import { useNavigate } from "react-router-dom";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";

function LandingPage() {
  const navigate = useNavigate();

  const handleStartQuiz = () => {
    navigate("/quiz");
  };

  return (
    <MKBox
      display="flex"
      justifyContent="center"
      alignItems="center"
      minHeight="80vh"
      flexDirection="column"
    >
      <MKTypography variant="h2" mb={2}>
        Discover the Perfect Gadget for You.
      </MKTypography>
      <MKButton variant="gradient" color="info" onClick={handleStartQuiz}>
        Take the Quick Quiz
      </MKButton>
    </MKBox>
  );
}

export default LandingPage;
