export default function Footer() {
    return (
      <footer className="mt-10 border-t">
        <div className="max-w-5xl mx-auto px-6 py-6 text-center">
          <div className="h-1 w-20 mx-auto mb-4 bg-gradient-to-r from-blue-500 to-teal-400 rounded-full"></div>
  
          <p className="text-sm text-gray-500">
            © {new Date().getFullYear()} Tiny Malaria Scan — Built with ❤️ by Avi
          </p>
        </div>
      </footer>
    );
  }
  