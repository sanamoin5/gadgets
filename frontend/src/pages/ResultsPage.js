import React, { useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import Card from "@mui/material/Card";
import CardMedia from "@mui/material/CardMedia";
import CardContent from "@mui/material/CardContent";
import axios from "axios"; // For API calls
import API_BASE_URL from "../config";

function ResultsPage() {
  const location = useLocation();
  const userAnswers = location.state || {};
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchRecommendations = async () => {
      try {
        // Format userAnswers to match the backend schema
        const formattedAnswers = Object.keys(userAnswers).map((key) => ({
          question_id: parseInt(key, 10),
          answer: userAnswers[key],
        }));

        console.log("Formatted Answers:", formattedAnswers);

        const response = await axios.post(`${API_BASE_URL}/results`, formattedAnswers);
        setRecommendations(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to load recommendations.");
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [userAnswers]);

  if (loading) return <MKTypography variant="h5">Loading...</MKTypography>;
  if (error) return <MKTypography variant="h5">{error}</MKTypography>;

  return (
    <MKBox
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="flex-start"
      minHeight="100vh"
      px={2}
      pt={12}
      sx={{
        boxSizing: "border-box",
        overflow: "auto",
      }}
    >
      <MKTypography variant="h4" mb={3}>
        Your Gadget Recommendations
      </MKTypography>
      {recommendations.map((gadget, index) => (
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
