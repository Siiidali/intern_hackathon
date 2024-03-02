import { Status } from '@prisma/client';

export interface ModelResponse {
  status: Status;
  cellular_component: number;
  molecular_function: number;
  biological_process: number;
}
