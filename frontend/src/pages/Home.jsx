import { Link } from "react-router-dom";
import toast from "react-hot-toast";
import usePageTitle from "../hooks/usePageTitle.js";

export default function Home() {
  usePageTitle("Tiny Malaria Scan | Home");

  const notifyStart = () =>
    toast.success("Welcome! Start scanning your RBC images.", {
      duration: 2000,
    });

  return (
    <div className="min-h-screen flex flex-col py-10 items-center justify-center text-center relative overflow-hidden px-4">
      <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-teal-50 -z-10"></div>
      <div className="absolute -top-20 -left-20 w-72 h-72 bg-blue-100 rounded-full blur-3xl opacity-40"></div>
      <div className="absolute -bottom-20 -right-20 w-72 h-72 bg-teal-100 rounded-full blur-3xl opacity-40"></div>

      <div className="max-w-3xl mx-auto space-y-5 md:space-y-6 animate-fade-in">

        <h1
          className="text-4xl md:text-6xl font-extrabold leading-tight tracking-tight"
          aria-label="AI Powered Malaria Cell Detection"
        >
          AI-Powered
          <span className="block text-blue-600 mt-1">
            Malaria Cell Detection
          </span>
        </h1>
        <p className="text-gray-600 text-lg md:text-xl">
          Upload your microscope RBC image and let our fine-tuned ResNet-18 model
          detect parasitized & uninfected cells with medical-grade precision.
        </p>
        <img
          src="/rbc-sample.png"
          alt="Microscopic RBC cell"
          className="mx-auto mt-6 w-40 h-40 md:w-30 opacity-90 rounded-3xl shadow-sm"
        />
        <div className="flex flex-col sm:flex-row justify-center gap-4 mt-6">

          <Link
            to="/scan"
            onClick={notifyStart}
            className="px-6 py-3 text-lg bg-blue-600 text-white rounded-lg 
            shadow-md hover:bg-blue-700 transition active:scale-95"
          >
            Start Scanning
          </Link>

          <Link
            to="/gradcam"
            className="px-6 py-3 text-lg border border-blue-600 text-blue-600 
            rounded-lg hover:bg-blue-50 transition active:scale-95"
          >
            View Grad-CAM
          </Link>

        </div>
      </div>
      <div className="mt-10 flex flex-wrap justify-center gap-3 opacity-80">
        <div className="px-4 py-2 bg-white rounded-md shadow text-sm border">
          ⚡ Real-time Inference
        </div>
        <div className="px-4 py-2 bg-white rounded-md shadow text-sm border">
          🔬 ResNet-18 CNN
        </div>
        <div className="px-4 py-2 bg-white rounded-md shadow text-sm border">
          🎯 High Accuracy
        </div>
        <div className="px-4 py-2 bg-white rounded-md shadow text-sm border">
          🧠 Grad-CAM Explainability
        </div>
      </div>

    </div>
  );
}
