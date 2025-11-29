import React from "react";

export default function PredictionResult({ prediction, confidence, uploadedImage }) {
  const confidencePercent = Math.round(confidence * 100);

  return (
    <div className="border rounded-xl shadow-md p-6 bg-white space-y-6">

      {/* Uploaded Image */}
      {uploadedImage && (
        <img
          src={uploadedImage}
          alt="Uploaded RBC"
          className="w-full max-h-64 object-contain border rounded-lg"
        />
      )}

      {/* Prediction */}
      <div>
        <p className="text-sm text-gray-500">Prediction</p>
        <h2 className="text-2xl font-bold text-gray-800">{prediction}</h2>
      </div>

      {/* Confidence */}
      <div>
        <p className="text-sm text-gray-500 mb-1">Confidence</p>

        <div className="w-full bg-gray-200 rounded-full h-3">
          <div
            className="bg-blue-600 h-3 rounded-full"
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>

        <p classname="text-sm font-semibold mt-1">{confidencePercent}%</p>
      </div>

      {/* Disclaimer */}
      <p className="text-xs text-gray-500 italic">
        This is an AI-assisted screening tool. Not a medical diagnosis.
      </p>
    </div>
  );
}
