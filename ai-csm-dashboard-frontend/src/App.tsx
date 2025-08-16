// src/App.tsx
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import DashboardPage from './pages/DashboardPage';
import UserDetailPage from './pages/UserDetailPage';
import CompanyDetailPage from './pages/CompanyDetailPage'; // <--- IMPORT NEW PAGE
import { Toaster } from "@/components/ui/sonner";

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-background font-sans antialiased">
        <main className="container mx-auto p-4 md:p-8">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/users/:userId" element={<UserDetailPage />} />
            <Route path="/companies/:companyId" element={<CompanyDetailPage />} /> {/* <--- ADD THIS ROUTE */}
          </Routes>
        </main>
        <Toaster />
      </div>
    </BrowserRouter>
  );
}

export default App;