import { Routes, Route, Navigate } from "react-router-dom";
import Login from "./pages/Login";

function App() {
  return (
    <>
      <Routes>
        {/* <Route
          path="/"
          element={user ? <Todos /> : <Navigate to={"/login"} />}
        /> */}
        <Route path="/login" element={<Login />} />
      </Routes>
    </>
  );
}

export default App;
