import { useMutation } from "react-query";
import {
  loginUser,
  registerUser,
  logoutUser,
  refreshTokens,
} from "../app/api/auth";

export const useLogin = () => {
  return useMutation({
    mutationFn: loginUser,
  });
};

export const useRegister = () => {
  return useMutation({
    mutationFn: registerUser,
  });
};

export const useLogout = () => {
  return useMutation({
    mutationFn: logoutUser,
  });
};

export const useRefreshTokens = () => {
  return useMutation({
    mutationFn: refreshTokens,
  });
};
