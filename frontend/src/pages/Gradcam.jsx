import { useState } from "react";

export default function Gradcam() {
  const [imageURL, setImageURL] = useState(null);
  const [gradcamURL, setGradcamURL] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [confidence, setConfidence] = useState(0);
  const [loading, setLoading] = useState(false);

  const analyzeGradcam = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("https://malaria-backend-gddz.onrender.com/gradcam", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Server error");
    }

    return res.json();
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageURL(URL.createObjectURL(file));
    setLoading(true);

    try {
      const result = await analyzeGradcam(file);

      setPrediction(result.prediction);
      setConfidence(result.confidence);
      setGradcamURL(`data:image/png;base64,${result.gradcam}`);
    } catch (err) {
      alert(err.message);
      setGradcamURL(null);
      setPrediction("");
      setConfidence(0);
    }

    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-10 space-y-8">

      {/* Page Title */}
      <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-teal-500 text-transparent bg-clip-text">
        Grad-CAM Heatmap Viewer
      </h1>

      <p className="text-gray-600">
        Visualize where the neural network focuses while detecting malaria parasites.
      </p>

      {/* Upload Input */}
      <label className="cursor-pointer block border-2 border-dashed border-gray-300 p-10 text-center rounded-lg hover:border-blue-400 transition">
        <p className="text-gray-600">Click to upload microscope image</p>
        <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
      </label>

      {/* Loading */}
      {loading && <p className="text-blue-600 font-medium">Generating heatmap...</p>}

      {/* Results */}
      {!loading && prediction && gradcamURL && (
        <div className="border rounded-xl shadow-lg p-6 bg-white space-y-6">

          {/* Horizontal layout */}
          <div className="grid md:grid-cols-2 gap-6">

            {/* Original image */}
            <div>
              <p className="text-sm text-gray-500 mb-1">Original Image</p>
              <img
                src={imageURL}
                alt="Uploaded RBC"
                className="rounded border w-full max-h-80 object-contain"
              />
            </div>

            {/* Grad-CAM image */}
            <div>
              <p className="text-sm text-gray-500 mb-1">Grad-CAM Heatmap</p>
              <img
                src={gradcamURL}
                alt="Heatmap"
                className="rounded border w-full max-h-80 object-contain"
              />
            </div>

          </div>

          {/* Prediction */}
          <div className="space-y-1">
            <p className="text-sm text-gray-500">Prediction</p>
            <h2 className="text-2xl font-bold">{prediction}</h2>
          </div>

          {/* Confidence */}
          <div>
            <p className="text-sm text-gray-500">Confidence</p>

            <div className="w-full bg-gray-200 h-3 rounded-full">
              <div
                className="h-3 rounded-full bg-blue-600"
                style={{ width: `${confidence}%` }}
              ></div>
            </div>

            <p className="mt-1 font-semibold">{confidence}%</p>
          </div>

        </div>
      )}

    </div>
  );
}
