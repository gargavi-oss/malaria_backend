import { BrowserRouter as Router, Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";

import Home from "./pages/Home";
import Scan from "./pages/Scan";
import About from "./pages/About";
import Gradcam from "./pages/Gradcam";
import { Toaster } from "react-hot-toast";


function Layout() {

<Toaster position="top-center" />
  const location = useLocation();

  const isHome = location.pathname === "/";

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />
      <main
        className={
          isHome
            ? "flex-1 p-0 m-0" 
            : "flex-1 container mx-auto px-4 py-6"
        }
      >
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/scan" element={<Scan />} />
          <Route path="/about" element={<About />} />
          <Route path="/gradcam" element={<Gradcam />} />
        </Routes>
      </main>

      <Footer />

    </div>
  );
}

export default function App() {
  return (
    <Router>
      <Layout />
    </Router>
  );
}
