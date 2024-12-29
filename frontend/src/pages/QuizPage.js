import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import quizData from "data/quizData"; // your dummy quiz data
import MKBox from "components/MKBox";
import MKTypography from "components/MKTypography";
import MKButton from "components/MKButton";

function QuizPage() {
  const navigate = useNavigate();
  const [currentStep, setCurrentStep] = useState(0);
  const [answers, setAnswers] = useState({});

  const handleNext = (selectedOption) => {
    setAnswers({ ...answers, [currentStep]: selectedOption });

    if (currentStep < quizData.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      // after the last question, go to results
      navigate("/results", { state: answers });
    }
  };

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
