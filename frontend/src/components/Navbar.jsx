import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();

  const isActive = (path) =>
    location.pathname === path ? "text-blue-600 font-semibold" : "text-gray-600";

  return (
    <header className="backdrop-blur-md bg-white/70 border-b shadow-sm sticky top-0 z-50">
      <nav className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
        
        {/* Logo */}
        <Link 
          to="/" 
          className="text-2xl font-extrabold bg-gradient-to-r from-blue-600 to-teal-500 bg-clip-text text-transparent"
        >
          Tiny Malaria Scan
        </Link>

        {/* Links */}
        <div className="flex gap-8 text-sm">
          <Link to="/" className={isActive("/")}>Home</Link>
          <Link to="/scan" className={isActive("/scan")}>Scan</Link>
          <Link to="/gradcam" className={isActive("/gradcam")}>Grad-CAM</Link>
          <Link to="/about" className={isActive("/about")}>About</Link>
        </div>
      </nav>
    </header>
  );
}
