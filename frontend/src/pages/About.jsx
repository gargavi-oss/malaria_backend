export default function AboutPage() {
    return (
      <div className="max-w-5xl mx-auto px-6 py-10">
  
        {/* Page Title */}
        <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-600 to-teal-500 text-transparent bg-clip-text">
          About Tiny Malaria Scan
        </h1>
  
        <p className="text-gray-600 leading-relaxed max-w-3xl">
          Tiny Malaria Scan is an AI-powered malaria detection tool that analyzes 
          microscopic RBC images using a fine-tuned ResNet-18 deep learning model.
          It is built to assist early screening — fast, lightweight, and explainable.
        </p>
  
        {/* Feature Grid */}
        <div className="mt-10 grid md:grid-cols-2 gap-6">
  
          <div className="p-6 border rounded-xl bg-white shadow hover:shadow-md transition">
            <h2 className="font-semibold text-lg mb-2">🔬 Technology</h2>
            <p className="text-gray-500 text-sm">
              Powered by PyTorch, FastAPI, React JS, and a custom-trained ResNet-18 model.
              Uses Grad-CAM for transparency in decision-making.
            </p>
          </div>
  
          <div className="p-6 border rounded-xl bg-white shadow hover:shadow-md transition">
            <h2 className="font-semibold text-lg mb-2">⚡ Fast & Lightweight</h2>
            <p className="text-gray-500 text-sm">
              Optimized for cloud + mobile. Model inference is extremely fast (under 200ms).
            </p>
          </div>
  
          <div className="p-6 border rounded-xl bg-white shadow hover:shadow-md transition">
            <h2 className="font-semibold text-lg mb-2">🧠 Explainable AI</h2>
            <p className="text-gray-500 text-sm">
              Grad-CAM heatmaps highlight infected regions inside RBC cells, improving clinician understanding.
            </p>
          </div>
  
          <div className="p-6 border rounded-xl bg-white shadow hover:shadow-md transition">
            <h2 className="font-semibold text-lg mb-2">👨‍💻 Created by Avi</h2>
            <p className="text-gray-500 text-sm">
              Built with passion for AI, healthcare, and full-stack development.
            </p>
          </div>
  
        </div>
  
        {/* Model Performance Section */}
        <h2 className="text-3xl font-bold mt-12 mb-6">Model Performance</h2>
  
        {/* Images Grid */}
        <div className="grid md:grid-cols-2 gap-8">
  
          {/* Confusion Matrix */}
          <div className="border rounded-xl p-5 bg-white shadow hover:shadow-md transition">
            <h3 className="font-semibold text-lg mb-3">📊 Confusion Matrix</h3>
            <img
              src="/cm.png"
              alt="Confusion Matrix"
              className="rounded border w-full object-contain max-h-80"
            />
            <p className="text-sm text-gray-500 mt-2">
              Shows classification accuracy for Parasitized vs Uninfected RBC cells.
            </p>
          </div>
  
          {/* ROC Curve */}
          <div className="border rounded-xl p-5 bg-white shadow hover:shadow-md transition">
            <h3 className="font-semibold text-lg mb-3">📈 ROC Curve</h3>
            <img
              src="/roc.png"
              alt="ROC Curve"
              className="rounded border w-full object-contain max-h-80"
            />
            <p className="text-sm text-gray-500 mt-2">
              High AUC indicates strong model performance in distinguishing infected cells.
            </p>
          </div>
  
        </div>
  
      </div>
    );
  }
  