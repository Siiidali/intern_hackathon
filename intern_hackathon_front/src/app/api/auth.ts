import axios from "axios";
const API_BASE_URL = "http://localhost:3000/api/v1"; // Replace with your actual API base URL

interface RegisterData {
  name: string;
  email: string;
  password: string;
}

interface LoginData {
  email: string;
  password: string;
}

interface RefreshTokenData {
  refreshToken: string;
}

// Function to register a new user
export const registerUser = async (userData: RegisterData) => {
  const response = await axios.post(`${API_BASE_URL}/auth/register`, userData);
  return response.data;
};

// Function to log in
export const loginUser = async (credentials: LoginData) => {
  const response = await axios.post(`${API_BASE_URL}/auth/login`, credentials);
  return response.data;
};

// Function to log out
export const logoutUser = async (refreshToken: RefreshTokenData) => {
  const response = await axios.post(`${API_BASE_URL}/auth/logout`, {
    refreshToken,
  });
  return response.data;
};

// Function to refresh tokens
export const refreshTokens = async (refreshToken: RefreshTokenData) => {
  const response = await axios.post(`${API_BASE_URL}/auth/refresh-tokens`, {
    refreshToken,
  });
  return response.data;
};
