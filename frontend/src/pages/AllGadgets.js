import React, { useEffect, useState } from "react";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import Card from "@mui/material/Card";
import CardMedia from "@mui/material/CardMedia";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import axios from "axios";
import API_BASE_URL from "../config";

function AllGadgetsPage() {
  const [gadgets, setGadgets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchGadgets = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/gadgets`); // Backend endpoint
        setGadgets(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to load gadgets.");
        setLoading(false);
      }
    };

    fetchGadgets();
  }, []);

  // Handle loading and error states
  if (loading) return <MKTypography variant="h5">Loading...</MKTypography>;
  if (error) return <MKTypography variant="h5">{error}</MKTypography>;

  return (
    <MKBox
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight="80vh"
      px={2}
      pt={{ xs: 10, sm: 12 }}
    >
      <MKTypography variant="h4" mb={3} textAlign="center">
        All Available Gadgets
      </MKTypography>
      <Grid container spacing={3} justifyContent="center">
        {gadgets.map((gadget) => (
          <Grid item xs={12} sm={6} md={4} key={gadget.id}>
            <Card>
              <CardMedia
                component="img"
                height="140"
                image={gadget.image}
                alt={gadget.name}
              />
              <CardContent>
                <MKTypography variant="h5" gutterBottom>
                  {gadget.name}
                </MKTypography>
                <MKTypography variant="body2" color="textSecondary">
                  {gadget.description}
                </MKTypography>
                <MKTypography variant="subtitle1" color="primary" mt={1}>
                  {gadget.price}
                </MKTypography>
              </CardContent>
            </Card>
          </Grid>
        ))}
      </Grid>
    </MKBox>
  );
}

export default AllGadgetsPage;
