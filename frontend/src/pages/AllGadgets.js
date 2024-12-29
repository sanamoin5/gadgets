import React from "react";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import Card from "@mui/material/Card";
import CardMedia from "@mui/material/CardMedia";
import CardContent from "@mui/material/CardContent";
import Grid from "@mui/material/Grid";
import recommendationData from "data/recommendationData";

function AllGadgetsPage() {
  return (
    <>
      <MKBox
        display="flex"
        flexDirection="column"
        alignItems="center"
        justifyContent="center"
        minHeight="80vh"
        px={2}
        pt={{ xs: 10, sm: 12 }} // Add padding-top to ensure content doesn't overlap with the navbar
      >
        <MKTypography variant="h4" mb={3} textAlign="center">
          All Available Gadgets
        </MKTypography>
        <Grid container spacing={3} justifyContent="center">
          {recommendationData.map((gadget, index) => (
            <Grid item xs={12} sm={6} md={4} key={index}>
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
    </>
  );
}

export default AllGadgetsPage;
