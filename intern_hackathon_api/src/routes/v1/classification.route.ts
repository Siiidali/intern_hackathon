import express from 'express';
// import validate from '../../middlewares/validate';
// import authValidation from '../../validations/auth.validation';
import { classificationController } from '../../controllers';
import auth from '../../middlewares/auth';
import checkPlan from '../../middlewares/checkUserPlan';
import uploadFasta from '../../middlewares/uploadFastaFiles';

const router = express.Router();

router.post('/', auth(), checkPlan, uploadFasta, classificationController.classifySequence);

export default router;
