import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";
import axios from "axios"; // For API calls
import API_BASE_URL from "../config";

function QuizPage() {
  const navigate = useNavigate();
  const [quizData, setQuizData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState({});
  const [loading, setLoading] = useState(true); // Loading state
  const [error, setError] = useState(null); // Error state

  // Fetch quiz data on component mount
  useEffect(() => {
    const fetchQuizData = async () => {
      try {
        const response = await axios.get(`${API_BASE_URL}/questions`); // Replace with your backend URL
        setQuizData(response.data); // Assuming response.data is an array of questions
        setLoading(false);
      } catch (err) {
        setError("Failed to load quiz data.");
        setLoading(false);
      }
    };

    fetchQuizData();
  }, []);

  const handleNext = (selectedOption) => {
    setAnswers({ ...answers, [currentStep]: selectedOption });

    if (currentStep < quizData.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // After the last question, navigate to results
      navigate("/results", { state: answers });
    }
  };

  // Handle loading and error states
  if (loading) return <MKTypography variant="h5">Loading...</MKTypography>;
  if (error) return <MKTypography variant="h5">{error}</MKTypography>;

  const questionData = quizData[currentStep];

  return (
    <MKBox
      display="flex"
      flexDirection="column"
      alignItems="center"
      justifyContent="center"
      minHeight="80vh"
    >
      <MKTypography variant="h5" mb={3}>
        {questionData.question}
      </MKTypography>
      {questionData.options.map((option) => (
        <MKButton
          key={option}
          onClick={() => handleNext(option)}
          variant="outlined"
          color="info"
          sx={{ mb: 2 }}
        >
          {option}
        </MKButton>
      ))}
      <MKTypography variant="button" mt={3}>
        Question {currentStep + 1} of {quizData.length}
      </MKTypography>
    </MKBox>
  );
}

export default QuizPage;
