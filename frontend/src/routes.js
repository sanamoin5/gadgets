import LandingPage from "./pages/LandingPage";
import QuizPage from "./pages/QuizPage";
import ResultsPage from "./pages/ResultsPage";
import AllGadgetsPage from "./pages/AllGadgets";
import Contact from "./pages/Contact";
import License from "./pages/License";

export const routes = [
  {
    name: "Home",
    route: "/",
    component: <LandingPage />,
    key: "home",
  },
  {
    name: "Take the Quiz",
    route: "/quiz",
    component: <QuizPage />,
    key: "quiz",
  },
  {
    name: "Results",
    route: "/results",
    component: <ResultsPage />,
    key: "results",
  },
  {
    name: "All Gadgets",
    route: "/all-gadgets",
    component: <AllGadgetsPage />,
    key: "all-gadgets",
  },
  {
    name: "Contact",
    route: "/contact",
    component: <Contact />,
    key: "contact",
  },
  {
    name: "License",
    route: "/license",
    component: <License />,
    key: "license",
  },
];

export const navLinks = [
  {
    name: "Home",
    route: "/",
    key: "nav-home",
  },
  {
    name: "Take the Quiz",
    route: "/quiz",
    key: "nav-quiz",
  },
  {
    name: "All Gadgets",
    route: "/all-gadgets",
    key: "nav-all-gadgets",
  },
];
