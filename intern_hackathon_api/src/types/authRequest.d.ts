import { Request } from 'express';
export interface AuthRequest extends Request {
  user?: User;
  file?: any;
  fastaFileName?: string;
}
