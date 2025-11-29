import { useState } from "react";
import PredictionResult from "../components/PredictionResult";

export default function Scan() {
  const [imageURL, setImageURL] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeImage = async (file) => {
    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch("https://malaria-backend-gddz.onrender.com/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) throw new Error((await res.json()).detail);

    return res.json();
  };

  const handleUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImageURL(URL.createObjectURL(file));
    setLoading(true);

    try {
      const data = await analyzeImage(file);
      setResult(data);
    } catch (err) {
      alert(err.message);
      setResult(null);
    }

    setLoading(false);
  };

  return (
    <div className="max-w-4xl mx-auto px-6 py-10">
      
      <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent">
        Scan RBC Cell
      </h1>

      <p className="text-gray-500 mb-6">
        Upload a microscope image of an RBC cell to detect malaria infection.
      </p>

      {/* Upload Area */}
      <label className="cursor-pointer block border-2 border-dashed border-gray-300 p-10 text-center rounded-lg hover:border-blue-400 transition">
        <p className="text-gray-600">Click to upload image</p>
        <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
      </label>

      {/* Loading */}
      {loading && (
        <p className="mt-4 text-blue-600 font-medium">Analyzing image…</p>
      )}

      {/* Result */}
      {result && imageURL && !loading && (
        <div className="mt-8">
          <PredictionResult
            prediction={result.prediction}
            confidence={result.confidence / 100}
            uploadedImage={imageURL}
          />
        </div>
      )}
    </div>
  );
}
