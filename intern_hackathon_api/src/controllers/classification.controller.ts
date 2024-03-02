import httpStatus from 'http-status';
import catchAsync from '../utils/catchAsync';
import { classificationService } from '../services';
import { AuthRequest } from '../types/authRequest';

const classifySequence = catchAsync(async (req: AuthRequest, res) => {
  const data = await classificationService.classifyProteinSequence(req.file);
  await classificationService.addProteinSequenceToUserHistory(
    req.user?.id,
    req.fastaFileName!,
    data
  );

  res.status(httpStatus.OK).send(data);
});

export default {
  classifySequence
};
