import React from "react";
import { useLocation } from "react-router-dom";
import recommendationData from "data/recommendationData"; //  dummy recommendations
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import Card from "@mui/material/Card";
import CardMedia from "@mui/material/CardMedia";
import CardContent from "@mui/material/CardContent";

function ResultsPage() {
  const location = useLocation();
  const userAnswers = location.state || {};

  return (
    <MKBox
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="flex-start" // Align content at the top
      minHeight="100vh" // Full viewport height
      px={2}
      pt={12} // Add padding to account for the navbar height
      sx={{
        boxSizing: "border-box", // Ensure padding is part of the box model
        overflow: "auto", // Handle content overflow gracefully
      }}
    >
      <MKTypography variant="h4" mb={3}>
        Your Gadget Recommendations
      </MKTypography>
      {recommendationData.map((gadget, index) => (
        <Card key={index} sx={{ maxWidth: 345, mb: 3 }}>
          <CardMedia
            component="img"
            height="140"
            image={gadget.image}
            alt={gadget.name}
          />
          <CardContent>
            <MKTypography variant="h5">{gadget.name}</MKTypography>
            <MKTypography variant="body2">{gadget.description}</MKTypography>
            <MKTypography variant="button" color="text" fontWeight="bold">
              {gadget.price}
            </MKTypography>
          </CardContent>
          <CardContent>
            <MKTypography variant="h5">{gadget.name}</MKTypography>
            <MKTypography variant="body2">{gadget.description}</MKTypography>
            <MKTypography variant="button" color="text" fontWeight="bold">
              {gadget.price}
            </MKTypography>
          </CardContent>
        </Card>
      ))}
      <MKBox mt={2}>
        <MKButton variant="outlined" color="info" href="/">
          Back to Home
        </MKButton>
      </MKBox>
    </MKBox>
  );
}

export default ResultsPage;
