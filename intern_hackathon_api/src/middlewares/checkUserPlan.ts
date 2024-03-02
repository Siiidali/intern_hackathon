import { User } from '@prisma/client';
import { NextFunction } from 'express';
import ApiError from '../utils/ApiError';
import { plans } from '../constants/plans';
import httpStatus from 'http-status';
import { userService } from '../services';
import { AuthRequest } from '../types/authRequest';

const checkPlan = () => async (req: AuthRequest, res: Response, next: NextFunction) => {
  const user = req.user as User;

  // Check if the user has a plan
  if (!user.planChoosed) {
    return next(new ApiError(httpStatus.FORBIDDEN, 'No plan chosen'));
  }

  // Check if the user has exceeded the maximum number of requests for the day
  const planChoosed = plans.find((plan) => plan.name === user.planChoosed);
  if (!planChoosed) {
    return next(new ApiError(httpStatus.FORBIDDEN, 'Invalid plan chosen'));
  }
  if (planChoosed.max_requests !== undefined) {
    const user = req.user as User;
    user.requests_made_by_day = user.requests_made_by_day ?? 0;
    if (user.requests_made_by_day >= planChoosed.max_requests) {
      return next(new ApiError(httpStatus.FORBIDDEN, 'Exceeded maximum requests for the day'));
    }

    const updatedUser = await userService.updateUserById(user.id, {
      requests_made_by_day: user.requests_made_by_day + 1
    });
    user.requests_made_by_day += 1;
  }

  next();
};

export default checkPlan;
