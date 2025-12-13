import { Disclosure, Transition } from "@headlessui/react";
import { Link, useLocation } from "react-router-dom";

export default function Navbar() {
  const location = useLocation();

  const navLinks = [
    { name: "Home", path: "/" },
    { name: "Scan", path: "/scan" },
    { name: "Grad-CAM", path: "/gradcam" },
    { name: "About", path: "/about" },
  ];

  return (
    <Disclosure as="nav" className="bg-white border-b shadow-sm sticky top-0 z-50">
      {({ open }) => (
        <>

          <div className="max-w-7xl mx-auto px-4 py-5 flex justify-between items-center">

            <Link to="/" className="text-xl font-bold text-blue-600">
              Tiny Malaria Scan
            </Link>


            <div className="hidden md:flex gap-6 text-sm font-medium">
              {navLinks.map(link => (
                <Link
                  key={link.path}
                  to={link.path}
                  className={`hover:text-blue-600 ${
                    location.pathname === link.path
                      ? "text-blue-600 font-semibold"
                      : "text-gray-700"
                  }`}
                >
                  {link.name}
                </Link>
              ))}
            </div>


            <Disclosure.Button className="md:hidden text-2xl focus:outline-none">
              {open ? "✕" : "☰"}
            </Disclosure.Button>
          </div>


          <Transition
            show={open}
            enter="transition ease-out duration-200"
            enterFrom="opacity-0 -translate-y-4"
            enterTo="opacity-100 translate-y-0"
            leave="transition ease-in duration-150"
            leaveFrom="opacity-100 translate-y-0"
            leaveTo="opacity-0 -translate-y-4"
          >
            <Disclosure.Panel className="md:hidden bg-white border-t shadow-sm">
              <div className="flex flex-col px-6 py-4 space-y-4 text-sm font-medium">
                {navLinks.map(link => (
                  <Link
                    key={link.path}
                    to={link.path}
                    className={`${
                      location.pathname === link.path
                        ? "text-blue-600 font-semibold"
                        : "text-gray-700"
                    }`}
                  >
                    {link.name}
                  </Link>
                ))}
              </div>
            </Disclosure.Panel>
          </Transition>
        </>
      )}
    </Disclosure>
  );
}
